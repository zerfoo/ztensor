package pjrt

import (
	"fmt"
	"unsafe"

	"github.com/zerfoo/ztensor/internal/cuda"
)

// Device wraps a PJRT_Device handle and provides methods for
// querying device properties: ID, kind, addressability, hardware ID.
//
// Device handles are owned by the Client and must not be destroyed
// independently — they become invalid when the Client is closed.
type Device struct {
	lib    *PJRTLib
	handle uintptr // PJRT_Device*
}

// ID returns the unique device ID within the client.
func (d *Device) ID() (int, error) {
	desc, err := d.getDescription()
	if err != nil {
		return 0, err
	}

	// PJRT_DeviceDescription_Id_Args:
	//   struct_size          uintptr
	//   device_description   uintptr
	//   id                   int64 (out)
	type idArgs struct {
		structSize        uintptr
		deviceDescription uintptr
		id                int64
	}
	args := idArgs{
		structSize:        unsafe.Sizeof(idArgs{}),
		deviceDescription: desc,
	}
	errPtr := cuda.Ccall(d.lib.PJRT_DeviceDescription_Id, uintptr(unsafe.Pointer(&args)))
	if err := d.lib.checkError(errPtr); err != nil {
		return 0, fmt.Errorf("PJRT_DeviceDescription_Id: %w", err)
	}
	return int(args.id), nil
}

// Kind returns the device kind string (e.g. "cpu", "gpu", "tpu").
func (d *Device) Kind() (string, error) {
	desc, err := d.getDescription()
	if err != nil {
		return "", err
	}

	// PJRT_DeviceDescription_Kind_Args:
	//   struct_size          uintptr
	//   device_description   uintptr
	//   device_kind          uintptr (out: const char*)
	//   device_kind_size     uintptr (out: size_t)
	type kindArgs struct {
		structSize        uintptr
		deviceDescription uintptr
		deviceKind        uintptr
		deviceKindSize    uintptr
	}
	args := kindArgs{
		structSize:        unsafe.Sizeof(kindArgs{}),
		deviceDescription: desc,
	}
	errPtr := cuda.Ccall(d.lib.PJRT_DeviceDescription_Kind, uintptr(unsafe.Pointer(&args)))
	if err := d.lib.checkError(errPtr); err != nil {
		return "", fmt.Errorf("PJRT_DeviceDescription_Kind: %w", err)
	}
	return goStringN(args.deviceKind, int(args.deviceKindSize)), nil
}

// IsAddressable returns true if this device can be directly accessed
// by the client (i.e. it is local to this process).
func (d *Device) IsAddressable() (bool, error) {
	// PJRT_Device_IsAddressable_Args:
	//   struct_size    uintptr
	//   device         uintptr
	//   is_addressable uint8 (out, C bool)
	type isAddressableArgs struct {
		structSize    uintptr
		device        uintptr
		isAddressable uint8
		_             [7]byte // padding to uintptr alignment
	}
	args := isAddressableArgs{
		structSize: unsafe.Sizeof(isAddressableArgs{}),
		device:     d.handle,
	}
	errPtr := cuda.Ccall(d.lib.PJRT_Device_IsAddressable, uintptr(unsafe.Pointer(&args)))
	if err := d.lib.checkError(errPtr); err != nil {
		return false, fmt.Errorf("PJRT_Device_IsAddressable: %w", err)
	}
	return args.isAddressable != 0, nil
}

// LocalHardwareId returns the hardware-level device ID.
// This is useful for multi-device systems (e.g. GPU index on a multi-GPU node).
func (d *Device) LocalHardwareId() (int, error) {
	// PJRT_Device_LocalHardwareId_Args:
	//   struct_size       uintptr
	//   device            uintptr
	//   local_hardware_id int32 (out)
	type localHWIDArgs struct {
		structSize      uintptr
		device          uintptr
		localHardwareId int32
		_               [4]byte // padding
	}
	args := localHWIDArgs{
		structSize: unsafe.Sizeof(localHWIDArgs{}),
		device:     d.handle,
	}
	errPtr := cuda.Ccall(d.lib.PJRT_Device_LocalHardwareId, uintptr(unsafe.Pointer(&args)))
	if err := d.lib.checkError(errPtr); err != nil {
		return 0, fmt.Errorf("PJRT_Device_LocalHardwareId: %w", err)
	}
	return int(args.localHardwareId), nil
}

// Handle returns the raw PJRT_Device pointer.
func (d *Device) Handle() uintptr {
	return d.handle
}

// getDescription calls PJRT_Device_GetDescription to obtain the
// PJRT_DeviceDescription pointer for this device.
func (d *Device) getDescription() (uintptr, error) {
	// PJRT_Device_GetDescription_Args:
	//   struct_size         uintptr
	//   device              uintptr
	//   device_description  uintptr (out)
	type getDescArgs struct {
		structSize        uintptr
		device            uintptr
		deviceDescription uintptr
	}
	args := getDescArgs{
		structSize: unsafe.Sizeof(getDescArgs{}),
		device:     d.handle,
	}
	errPtr := cuda.Ccall(d.lib.PJRT_Device_GetDescription, uintptr(unsafe.Pointer(&args)))
	if err := d.lib.checkError(errPtr); err != nil {
		return 0, fmt.Errorf("PJRT_Device_GetDescription: %w", err)
	}
	if args.deviceDescription == 0 {
		return 0, fmt.Errorf("pjrt: PJRT_Device_GetDescription returned null")
	}
	return args.deviceDescription, nil
}
