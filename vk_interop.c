/**
 * vk_interop.c — Minimal Vulkan-CUDA interop for zero-copy display.
 *
 * Exposes to Python:
 *   init(width, height, title) → cuda_device_ptr (int)
 *   present()                  → None
 *   cleanup()                  → None
 *   poll_events()              → list of event dicts
 *
 * CUDA writes BGRA8 pixels into the shared buffer, present() copies
 * the buffer into a swapchain image and displays it.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>

/* ── Globals ────────────────────────────────────────────────────── */

static SDL_Window*       g_window;
static VkInstance        g_instance;
static VkSurfaceKHR     g_surface;
static VkPhysicalDevice  g_phys_dev;
static VkDevice          g_device;
static VkQueue           g_queue;
static uint32_t          g_queue_family;
static VkSwapchainKHR    g_swapchain;
static VkFormat          g_swapchain_fmt;
static VkExtent2D        g_extent;
static uint32_t          g_image_count;
static VkImage*          g_swapchain_images;
static VkCommandPool     g_cmd_pool;
static VkCommandBuffer   g_cmd_buf;
static VkSemaphore       g_sem_available;
static VkSemaphore       g_sem_finished;
static VkFence           g_fence;

/* Shared buffer */
static VkBuffer          g_buffer;
static VkDeviceMemory    g_vk_memory;
static VkDeviceSize      g_alloc_size;
static cudaExternalMemory_t g_ext_mem;
static void*             g_cuda_ptr;
static int               g_width, g_height;

/* Function pointers for extensions */
static PFN_vkGetMemoryFdKHR pfn_vkGetMemoryFdKHR;

/* ── Helpers ────────────────────────────────────────────────────── */

#define VK_CHECK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        PyErr_Format(PyExc_RuntimeError, "%s failed: %d", #call, _r); \
        return NULL; \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        PyErr_Format(PyExc_RuntimeError, "%s failed: %s", #call, cudaGetErrorString(_e)); \
        return NULL; \
    } \
} while(0)

static uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(g_phys_dev, &mem_props);
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_props.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    return UINT32_MAX;
}

/* ── init() ─────────────────────────────────────────────────────── */

static PyObject* vk_init(PyObject* self, PyObject* args) {
    const char* title = "Warp Fire (Vulkan)";
    if (!PyArg_ParseTuple(args, "ii|s", &g_width, &g_height, &title))
        return NULL;

    /* SDL2 */
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        PyErr_Format(PyExc_RuntimeError, "SDL_Init: %s", SDL_GetError());
        return NULL;
    }
    g_window = SDL_CreateWindow(title,
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        g_width, g_height, SDL_WINDOW_VULKAN | SDL_WINDOW_SHOWN);
    if (!g_window) {
        PyErr_Format(PyExc_RuntimeError, "SDL_CreateWindow: %s", SDL_GetError());
        return NULL;
    }

    /* Instance */
    unsigned int sdl_ext_count;
    SDL_Vulkan_GetInstanceExtensions(g_window, &sdl_ext_count, NULL);
    const char** inst_exts = malloc((sdl_ext_count + 1) * sizeof(char*));
    SDL_Vulkan_GetInstanceExtensions(g_window, &sdl_ext_count, inst_exts);
    inst_exts[sdl_ext_count] = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;

    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "warp-fire",
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo inst_ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = sdl_ext_count + 1,
        .ppEnabledExtensionNames = inst_exts,
    };
    VK_CHECK(vkCreateInstance(&inst_ci, NULL, &g_instance));
    free(inst_exts);

    /* Surface */
    if (!SDL_Vulkan_CreateSurface(g_window, g_instance, &g_surface)) {
        PyErr_Format(PyExc_RuntimeError, "SDL_Vulkan_CreateSurface: %s", SDL_GetError());
        return NULL;
    }

    /* Physical device — pick first with graphics + present */
    uint32_t dev_count = 0;
    vkEnumeratePhysicalDevices(g_instance, &dev_count, NULL);
    VkPhysicalDevice* devs = malloc(dev_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(g_instance, &dev_count, devs);
    g_phys_dev = devs[0]; /* TODO: pick NVIDIA specifically */
    free(devs);

    /* Queue family */
    uint32_t qf_count;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys_dev, &qf_count, NULL);
    VkQueueFamilyProperties* qf_props = malloc(qf_count * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys_dev, &qf_count, qf_props);
    g_queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qf_count; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_phys_dev, i, g_surface, &present);
        if ((qf_props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            g_queue_family = i;
            break;
        }
    }
    free(qf_props);
    if (g_queue_family == UINT32_MAX) {
        PyErr_SetString(PyExc_RuntimeError, "No suitable queue family");
        return NULL;
    }

    /* Logical device */
    float prio = 1.0f;
    VkDeviceQueueCreateInfo queue_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = g_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &prio,
    };
    const char* dev_exts[] = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dev_ci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_ci,
        .enabledExtensionCount = 3,
        .ppEnabledExtensionNames = dev_exts,
    };
    VK_CHECK(vkCreateDevice(g_phys_dev, &dev_ci, NULL, &g_device));
    vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);

    /* Load extension function */
    pfn_vkGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)
        vkGetDeviceProcAddr(g_device, "vkGetMemoryFdKHR");
    if (!pfn_vkGetMemoryFdKHR) {
        PyErr_SetString(PyExc_RuntimeError, "vkGetMemoryFdKHR not available");
        return NULL;
    }

    /* Swapchain */
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_phys_dev, g_surface, &caps);
    g_extent = (caps.currentExtent.width != UINT32_MAX)
        ? caps.currentExtent
        : (VkExtent2D){g_width, g_height};

    /* Pick format */
    uint32_t fmt_count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_phys_dev, g_surface, &fmt_count, NULL);
    VkSurfaceFormatKHR* fmts = malloc(fmt_count * sizeof(VkSurfaceFormatKHR));
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_phys_dev, g_surface, &fmt_count, fmts);
    g_swapchain_fmt = fmts[0].format;
    VkColorSpaceKHR color_space = fmts[0].colorSpace;
    for (uint32_t i = 0; i < fmt_count; i++) {
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM) {
            g_swapchain_fmt = fmts[i].format;
            color_space = fmts[i].colorSpace;
            break;
        }
    }
    free(fmts);

    /* Pick present mode — prefer MAILBOX (no vsync, no tearing), else IMMEDIATE */
    uint32_t pm_count;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_phys_dev, g_surface, &pm_count, NULL);
    VkPresentModeKHR* pmodes = malloc(pm_count * sizeof(VkPresentModeKHR));
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_phys_dev, g_surface, &pm_count, pmodes);
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    for (uint32_t i = 0; i < pm_count; i++) {
        if (pmodes[i] == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
        if (pmodes[i] == VK_PRESENT_MODE_IMMEDIATE_KHR)
            present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    free(pmodes);

    uint32_t img_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && img_count > caps.maxImageCount)
        img_count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR sc_ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = g_surface,
        .minImageCount = img_count,
        .imageFormat = g_swapchain_fmt,
        .imageColorSpace = color_space,
        .imageExtent = g_extent,
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = present_mode,
        .clipped = VK_TRUE,
    };
    VK_CHECK(vkCreateSwapchainKHR(g_device, &sc_ci, NULL, &g_swapchain));

    vkGetSwapchainImagesKHR(g_device, g_swapchain, &g_image_count, NULL);
    g_swapchain_images = malloc(g_image_count * sizeof(VkImage));
    vkGetSwapchainImagesKHR(g_device, g_swapchain, &g_image_count, g_swapchain_images);

    /* Command pool + buffer */
    VkCommandPoolCreateInfo pool_ci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = g_queue_family,
    };
    VK_CHECK(vkCreateCommandPool(g_device, &pool_ci, NULL, &g_cmd_pool));

    VkCommandBufferAllocateInfo cb_ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = g_cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VK_CHECK(vkAllocateCommandBuffers(g_device, &cb_ai, &g_cmd_buf));

    /* Sync objects */
    VkSemaphoreCreateInfo sem_ci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(g_device, &sem_ci, NULL, &g_sem_available));
    VK_CHECK(vkCreateSemaphore(g_device, &sem_ci, NULL, &g_sem_finished));
    VkFenceCreateInfo fence_ci = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    VK_CHECK(vkCreateFence(g_device, &fence_ci, NULL, &g_fence));

    /* ── Shared buffer (Vulkan → CUDA) ────────────────────────── */
    VkDeviceSize buf_size = (VkDeviceSize)g_width * g_height * 4; /* BGRA8 */

    VkExternalMemoryBufferCreateInfo ext_buf_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VkBufferCreateInfo buf_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext_buf_info,
        .size = buf_size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    VK_CHECK(vkCreateBuffer(g_device, &buf_ci, NULL, &g_buffer));

    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(g_device, g_buffer, &mem_reqs);
    g_alloc_size = mem_reqs.size;

    VkExportMemoryAllocateInfo export_info = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    uint32_t mem_type = find_memory_type(mem_reqs.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mem_type == UINT32_MAX) {
        PyErr_SetString(PyExc_RuntimeError, "No suitable DEVICE_LOCAL memory type");
        return NULL;
    }
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &export_info,
        .allocationSize = g_alloc_size,
        .memoryTypeIndex = mem_type,
    };
    VK_CHECK(vkAllocateMemory(g_device, &alloc_info, NULL, &g_vk_memory));
    VK_CHECK(vkBindBufferMemory(g_device, g_buffer, g_vk_memory, 0));

    /* Export FD */
    int fd;
    VkMemoryGetFdInfoKHR fd_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = g_vk_memory,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
    };
    VK_CHECK(pfn_vkGetMemoryFdKHR(g_device, &fd_info, &fd));

    /* CUDA import */
    struct cudaExternalMemoryHandleDesc ext_mem_desc;
    memset(&ext_mem_desc, 0, sizeof(ext_mem_desc));
    ext_mem_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    ext_mem_desc.handle.fd = fd;
    ext_mem_desc.size = g_alloc_size;
    CUDA_CHECK(cudaImportExternalMemory(&g_ext_mem, &ext_mem_desc));
    /* fd is now owned by CUDA, do not close */

    struct cudaExternalMemoryBufferDesc cuda_buf_desc;
    memset(&cuda_buf_desc, 0, sizeof(cuda_buf_desc));
    cuda_buf_desc.offset = 0;
    cuda_buf_desc.size = buf_size;
    cuda_buf_desc.flags = 0;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&g_cuda_ptr, g_ext_mem, &cuda_buf_desc));

    return PyLong_FromUnsignedLongLong((unsigned long long)g_cuda_ptr);
}

/* ── present() ──────────────────────────────────────────────────── */

static PyObject* vk_present(PyObject* self, PyObject* args) {
    /* Wait for previous frame */
    vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, UINT64_MAX);
    vkResetFences(g_device, 1, &g_fence);

    /* Acquire swapchain image */
    uint32_t img_idx;
    VkResult acq = vkAcquireNextImageKHR(g_device, g_swapchain, UINT64_MAX,
        g_sem_available, VK_NULL_HANDLE, &img_idx);
    if (acq == VK_ERROR_OUT_OF_DATE_KHR || acq == VK_SUBOPTIMAL_KHR) {
        /* TODO: recreate swapchain */
        Py_RETURN_NONE;
    }

    /* Record command buffer: transition image, copy buffer→image, transition for present */
    vkResetCommandBuffer(g_cmd_buf, 0);
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(g_cmd_buf, &begin_info);

    /* Transition: UNDEFINED → TRANSFER_DST */
    VkImageMemoryBarrier barrier1 = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = g_swapchain_images[img_idx],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(g_cmd_buf,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier1);

    /* Copy buffer → image */
    VkBufferImageCopy region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {g_extent.width, g_extent.height, 1},
    };
    vkCmdCopyBufferToImage(g_cmd_buf, g_buffer, g_swapchain_images[img_idx],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    /* Transition: TRANSFER_DST → PRESENT_SRC */
    VkImageMemoryBarrier barrier2 = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = g_swapchain_images[img_idx],
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    vkCmdPipelineBarrier(g_cmd_buf,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, NULL, 0, NULL, 1, &barrier2);

    vkEndCommandBuffer(g_cmd_buf);

    /* Submit */
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &g_sem_available,
        .pWaitDstStageMask = &wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &g_cmd_buf,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &g_sem_finished,
    };
    vkQueueSubmit(g_queue, 1, &submit_info, g_fence);

    /* Present */
    VkPresentInfoKHR present_info = {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &g_sem_finished,
        .swapchainCount = 1,
        .pSwapchains = &g_swapchain,
        .pImageIndices = &img_idx,
    };
    vkQueuePresentKHR(g_queue, &present_info);

    Py_RETURN_NONE;
}

/* ── poll_events() ──────────────────────────────────────────────── */

static PyObject* vk_poll_events(PyObject* self, PyObject* args) {
    PyObject* events = PyList_New(0);
    SDL_Event ev;
    while (SDL_PollEvent(&ev)) {
        if (ev.type == SDL_QUIT) {
            PyObject* d = Py_BuildValue("{s:s}", "type", "quit");
            PyList_Append(events, d);
            Py_DECREF(d);
        } else if (ev.type == SDL_KEYDOWN) {
            PyObject* d = Py_BuildValue("{s:s,s:i}",
                "type", "keydown", "key", ev.key.keysym.sym);
            PyList_Append(events, d);
            Py_DECREF(d);
        }
    }
    return events;
}

/* ── set_title() ────────────────────────────────────────────────── */

static PyObject* vk_set_title(PyObject* self, PyObject* args) {
    const char* title;
    if (!PyArg_ParseTuple(args, "s", &title))
        return NULL;
    SDL_SetWindowTitle(g_window, title);
    Py_RETURN_NONE;
}

/* ── cleanup() ──────────────────────────────────────────────────── */

static PyObject* vk_cleanup(PyObject* self, PyObject* args) {
    vkDeviceWaitIdle(g_device);

    if (g_ext_mem) cudaDestroyExternalMemory(g_ext_mem);
    if (g_fence) vkDestroyFence(g_device, g_fence, NULL);
    if (g_sem_finished) vkDestroySemaphore(g_device, g_sem_finished, NULL);
    if (g_sem_available) vkDestroySemaphore(g_device, g_sem_available, NULL);
    if (g_cmd_pool) vkDestroyCommandPool(g_device, g_cmd_pool, NULL);
    if (g_buffer) vkDestroyBuffer(g_device, g_buffer, NULL);
    if (g_vk_memory) vkFreeMemory(g_device, g_vk_memory, NULL);
    free(g_swapchain_images);
    if (g_swapchain) vkDestroySwapchainKHR(g_device, g_swapchain, NULL);
    if (g_device) vkDestroyDevice(g_device, NULL);
    if (g_surface) vkDestroySurfaceKHR(g_instance, g_surface, NULL);
    if (g_instance) vkDestroyInstance(g_instance, NULL);
    if (g_window) SDL_DestroyWindow(g_window);
    SDL_Quit();

    Py_RETURN_NONE;
}

/* ── Module definition ──────────────────────────────────────────── */

static PyMethodDef methods[] = {
    {"init", vk_init, METH_VARARGS, "init(w, h, [title]) -> cuda_ptr"},
    {"present", vk_present, METH_NOARGS, "Copy shared buffer to screen"},
    {"poll_events", vk_poll_events, METH_NOARGS, "Poll SDL events"},
    {"set_title", vk_set_title, METH_VARARGS, "Set window title"},
    {"cleanup", vk_cleanup, METH_NOARGS, "Destroy all resources"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "vk_interop", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit_vk_interop(void) {
    return PyModule_Create(&module);
}
