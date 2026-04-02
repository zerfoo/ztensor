package pjrt

import (
	"fmt"
	"unsafe"
)

// Client wraps a PJRT_Client handle and provides Go-friendly methods
// for querying the runtime: platform name/version, device enumeration.
type Client struct {
	lib    *PJRTLib
	handle uintptr // PJRT_Client*
}

// ClientOption configures client creation.
type ClientOption func(*clientConfig)

type clientConfig struct {
	// createOptions is an opaque pointer to plugin-specific options.
	// For most plugins this is nil (use defaults).
	createOptions uintptr
}

// WithCreateOptions sets plugin-specific creation options.
func WithCreateOptions(opts uintptr) ClientOption {
	return func(c *clientConfig) {
		c.createOptions = opts
	}
}

// NewClient creates a PJRT client using the given plugin library.
// The client must be closed with Close() when no longer needed.
func NewClient(lib *PJRTLib, opts ...ClientOption) (*Client, error) {
	if lib == nil || lib.handle == 0 {
		return nil, fmt.Errorf("pjrt: cannot create client from nil or closed library")
	}

	var cfg clientConfig
	for _, o := range opts {
		o(&cfg)
	}

	// PJRT_Client_Create_Args:
	//   struct_size     uintptr
	//   create_options  uintptr  (plugin-specific, may be 0)
	//   client          uintptr  (out: PJRT_Client*)
	type createArgs struct {
		structSize    uintptr
		createOptions uintptr
		client        uintptr
	}

	args := createArgs{
		structSize:    unsafe.Sizeof(createArgs{}),
		createOptions: cfg.createOptions,
	}

	errPtr := ccall(lib.PJRT_Client_Create, uintptr(unsafe.Pointer(&args)))
	if err := lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Client_Create: %w", err)
	}
	if args.client == 0 {
		return nil, fmt.Errorf("pjrt: PJRT_Client_Create returned null client")
	}

	return &Client{lib: lib, handle: args.client}, nil
}

// Close destroys the PJRT client and releases associated resources.
// Safe to call multiple times.
func (c *Client) Close() error {
	if c.handle == 0 {
		return nil
	}

	type destroyArgs struct {
		structSize uintptr
		client     uintptr
	}
	args := destroyArgs{
		structSize: unsafe.Sizeof(destroyArgs{}),
		client:     c.handle,
	}
	errPtr := ccall(c.lib.PJRT_Client_Destroy, uintptr(unsafe.Pointer(&args)))
	c.handle = 0
	return c.lib.checkError(errPtr)
}

// PlatformName returns the name of the platform (e.g. "cpu", "cuda", "tpu").
func (c *Client) PlatformName() (string, error) {
	// PJRT_Client_PlatformName_Args:
	//   struct_size          uintptr
	//   client               uintptr
	//   platform_name        uintptr  (out: const char*)
	//   platform_name_size   uintptr  (out: size_t)
	type platformNameArgs struct {
		structSize       uintptr
		client           uintptr
		platformName     uintptr
		platformNameSize uintptr
	}
	args := platformNameArgs{
		structSize: unsafe.Sizeof(platformNameArgs{}),
		client:     c.handle,
	}
	errPtr := ccall(c.lib.PJRT_Client_PlatformName, uintptr(unsafe.Pointer(&args)))
	if err := c.lib.checkError(errPtr); err != nil {
		return "", fmt.Errorf("PJRT_Client_PlatformName: %w", err)
	}
	return goStringN(args.platformName, int(args.platformNameSize)), nil
}

// PlatformVersion returns the version string of the platform.
func (c *Client) PlatformVersion() (string, error) {
	type platformVersionArgs struct {
		structSize          uintptr
		client              uintptr
		platformVersion     uintptr
		platformVersionSize uintptr
	}
	args := platformVersionArgs{
		structSize: unsafe.Sizeof(platformVersionArgs{}),
		client:     c.handle,
	}
	errPtr := ccall(c.lib.PJRT_Client_PlatformVersion, uintptr(unsafe.Pointer(&args)))
	if err := c.lib.checkError(errPtr); err != nil {
		return "", fmt.Errorf("PJRT_Client_PlatformVersion: %w", err)
	}
	return goStringN(args.platformVersion, int(args.platformVersionSize)), nil
}

// Devices returns all devices known to the client (including non-addressable).
func (c *Client) Devices() ([]*Device, error) {
	// PJRT_Client_Devices_Args:
	//   struct_size  uintptr
	//   client       uintptr
	//   devices      uintptr  (out: PJRT_Device** array pointer)
	//   num_devices  uintptr  (out: size_t)
	type devicesArgs struct {
		structSize uintptr
		client     uintptr
		devices    uintptr
		numDevices uintptr
	}
	args := devicesArgs{
		structSize: unsafe.Sizeof(devicesArgs{}),
		client:     c.handle,
	}
	errPtr := ccall(c.lib.PJRT_Client_Devices, uintptr(unsafe.Pointer(&args)))
	if err := c.lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Client_Devices: %w", err)
	}
	return c.wrapDevices(args.devices, int(args.numDevices)), nil
}

// AddressableDevices returns devices that this client can directly interact with.
func (c *Client) AddressableDevices() ([]*Device, error) {
	type addressableDevicesArgs struct {
		structSize uintptr
		client     uintptr
		devices    uintptr
		numDevices uintptr
	}
	args := addressableDevicesArgs{
		structSize: unsafe.Sizeof(addressableDevicesArgs{}),
		client:     c.handle,
	}
	errPtr := ccall(c.lib.PJRT_Client_AddressableDevices, uintptr(unsafe.Pointer(&args)))
	if err := c.lib.checkError(errPtr); err != nil {
		return nil, fmt.Errorf("PJRT_Client_AddressableDevices: %w", err)
	}
	return c.wrapDevices(args.devices, int(args.numDevices)), nil
}

// wrapDevices converts a C array of PJRT_Device* pointers into Go Device structs.
//
//go:nocheckptr
func (c *Client) wrapDevices(arrayPtr uintptr, n int) []*Device {
	if n == 0 || arrayPtr == 0 {
		return nil
	}
	// arrayPtr is a PJRT_Device** — an array of n pointers.
	ptrs := unsafe.Slice((*uintptr)(unsafe.Pointer(arrayPtr)), n)
	devices := make([]*Device, n)
	for i, ptr := range ptrs {
		devices[i] = &Device{lib: c.lib, handle: ptr}
	}
	return devices
}

// Handle returns the raw PJRT_Client pointer for use by other PJRT wrappers.
func (c *Client) Handle() uintptr {
	return c.handle
}

// Lib returns the PJRTLib associated with this client.
func (c *Client) Lib() *PJRTLib {
	return c.lib
}
