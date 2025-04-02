// Repository: djthorpe/gopi
// File: hw.go

package gopi

import (
	"fmt"
	"strings"
	"time"
)

/*
	This file contains interface defininitons for hardware implementations:

	* Information about the underlying hardware platform
	* Pixel-based displays
	* SPI, I2C and GPIO
	* Infrared sending and receiving
*/

////////////////////////////////////////////////////////////////////////////////
// TYPES

type (
	PlatformType uint32
	DisplayFlag  uint32
	SPIMode      uint32 // SPIMode is the SPI Mode
	GPIOPin      uint8  // GPIOPin is the logical GPIO pin
	GPIOState    uint8  // GPIOState is the GPIO Pin state
	GPIOMode     uint8  // GPIOMode is the GPIO Pin mode
	GPIOPull     uint8  // GPIOPull is the GPIO Pin resistor configuration (pull up/down or floating)
	GPIOEdge     uint8  // GPIOEdge is a rising or falling edge
	LIRCMode     uint32 // LIRCMode is the LIRC Mode
	LIRCType     uint32 // LIRCType is the LIRC Type
)

type SPIBus struct {
	Bus, Slave uint
}

type I2CBus uint

////////////////////////////////////////////////////////////////////////////////
// INTERFACES

type Platform interface {
	Product() string                           // Product returns product name
	Type() PlatformType                        // Type returns flags identifying platform type
	SerialNumber() string                      // SerialNumber returns unique serial number for host
	Uptime() time.Duration                     // Uptime returns uptime for host
	LoadAverages() (float64, float64, float64) // LoadAverages returns 1, 5 and 15 minute load averages
	TemperatureZones() map[string]float32      // Return celcius values for zones
}

// DisplayManager manages the connected displays and emits Display objects
// when their state changes
type DisplayManager interface {
	// Displays return all attached displays
	Displays() []Display

	// Displays returns specific display
	Display(uint32) (Display, error)

	// PowerOn a display, can return ErrNotImplemented if
	// a specific display does not support control
	PowerOn(Display) error

	// PowerOff a display, can return ErrNotImplemented if
	// a specific display does not support control
	PowerOff(Display) error
}

// Display implements a pixel-based display device
type Display interface {
	Id() uint32             // Return display number
	Flags() DisplayFlag     // Return information about the display
	Name() string           // Return name of the display
	Size() (uint32, uint32) // Return display size for nominated display number, or (0,0) if display does not exist
	PixelsPerInch() uint32  // Return the PPI (pixels-per-inch) for the display, or return zero if unknown
}

// SPI implements the SPI interface for sensors, etc.
type SPI interface {
	// Return all valid devices
	Devices() []SPIBus

	// Mode returns the current SPI mode
	Mode(SPIBus) SPIMode

	// SetMode sets the current SPI mode
	SetMode(SPIBus, SPIMode) error

	// MaxSpeedHz returns the current data transfer speed
	MaxSpeedHz(SPIBus) uint32

	// SetMaxSpeedHz sets the current data transfer speed
	SetMaxSpeedHz(SPIBus, uint32) error

	// BitsPerWord returns current configuration
	BitsPerWord(SPIBus) uint8

	// SetBitsPerWord updates the current configuration
	SetBitsPerWord(SPIBus, uint8) error

	// Transfer reads and writes on the SPI bus
	Transfer(SPIBus, []byte) ([]byte, error)

	// Read bytes into a buffer
	Read(SPIBus, []byte) error

	// Write bytes from the buffer
	Write(SPIBus, []byte) error
}

// I2C implements the I2C interface for sensors, etc.
type I2C interface {
	// Return all valid devices
	Devices() []I2CBus

	// Set current slave address
	SetSlave(I2CBus, uint8) error

	// Get current slave address
	GetSlave(I2CBus) uint8

	// Return true if a slave was detected at a particular address
	DetectSlave(I2CBus, uint8) (bool, error)

	// Read and Write data directly
	Read(I2CBus) ([]byte, error)
	Write(I2CBus, []byte) (int, error)

	// Read Byte (8-bits), Word (16-bits) & Block ([]byte) from registers
	ReadUint8(bus I2CBus, reg uint8) (uint8, error)
	ReadInt8(bus I2CBus, reg uint8) (int8, error)
	ReadUint16(bus I2CBus, reg uint8) (uint16, error)
	ReadInt16(bus I2CBus, reg uint8) (int16, error)
	ReadBlock(bus I2CBus, reg, length uint8) ([]byte, error)

	// Write Byte (8-bits) and Word (16-bits)
	WriteUint8(bus I2CBus, reg, value uint8) error
	WriteInt8(bus I2CBus, reg uint8, value int8) error
	WriteUint16(bus I2CBus, reg uint8, value uint16) error
	WriteInt16(bus I2CBus, reg uint8, value int16) error
}

// GPIO implements the GPIO interface for simple input and output
type GPIO interface {
	// Return number of physical pins, or 0 if if cannot be returned
	// or nothing is known about physical pins
	NumberOfPhysicalPins() uint

	// Return array of available logical pins or nil if nothing is
	// known about pins
	Pins() []GPIOPin

	// Return logical pin for physical pin number. Returns
	// GPIO_PIN_NONE where there is no logical pin at that position
	// or we don't now about the physical pins
	PhysicalPin(uint) GPIOPin

	// Return physical pin number for logical pin. Returns 0 where there
	// is no physical pin for this logical pin, or we don't know anything
	// about the layout
	PhysicalPinForPin(GPIOPin) uint

	// Read pin state
	ReadPin(GPIOPin) GPIOState

	// Write pin state
	WritePin(GPIOPin, GPIOState)

	// Get pin mode
	GetPinMode(GPIOPin) GPIOMode

	// Set pin mode
	SetPinMode(GPIOPin, GPIOMode)

	// Set pull mode to pull down or pull up - will
	// return ErrNotImplemented if not supported
	SetPullMode(GPIOPin, GPIOPull) error

	// Start watching for rising and/or falling edge,
	// or stop watching when GPIO_EDGE_NONE is passed.
	// Will return ErrNotImplemented if not supported
	Watch(GPIOPin, GPIOEdge) error
}

// GPIOEvent happens when a pin is watched and edge is
// either rising or falling
type GPIOEvent interface {
	Event
	Pin() GPIOPin
	Edge() GPIOEdge
}

// LIRC implements the IR send & receive interface
type LIRC interface {
	// Get receive and send modes
	RecvMode() LIRCMode
	SendMode() LIRCMode
	SetRecvMode(LIRCMode) error
	SetSendMode(LIRCMode) error

	// Receive parameters
	RecvDutyCycle() uint32
	RecvResolutionMicros() uint32

	// Receive parameters
	SetRecvTimeoutMs(uint32) error
	SetRecvTimeoutReports(bool) error
	SetRecvCarrierHz(uint32) error
	SetRecvCarrierRangeHz(min, max uint32) error

	// Send parameters
	SendDutyCycle() uint32
	SetSendCarrierHz(uint32) error
	SetSendDutyCycle(uint32) error

	// Send Pulse Mode, values are in milliseconds
	PulseSend([]uint32) error
}

// LIRCEvent is a pulse, space or timeout from an IR sensor
type LIRCEvent interface {
	Event
	Type() LIRCType
	Mode() LIRCMode
	Value() interface{} // value is uint32 in ms when mode is LIRC_MODE_MODE2
}

// LIRCKeycodeManager manages the database of keycodes and IR codes
type LIRCKeycodeManager interface {
	// Keycode returns keycodes which match a search phrase
	Keycode(string) []KeyCode

	// Lookup returns Keycodes in priority order for scancode
	Lookup(InputDevice, uint32) []KeyCode

	// Set Keycode for scancode, InputDevice and device name
	Set(InputDevice, uint32, KeyCode, string) error
}

////////////////////////////////////////////////////////////////////////////////
// CONSTANTS

const (
	PLATFORM_NONE PlatformType = 0
	// OS
	PLATFORM_DARWIN PlatformType = (1 << iota) >> 1
	PLATFORM_RPI
	PLATFORM_LINUX
	// CPU
	PLATFORM_X86_32
	PLATFORM_X86_64
	PLATFORM_BCM2835_ARM6
	PLATFORM_BCM2836_ARM7
	PLATFORM_BCM2837_ARM8
	PLATFORM_BCM2838_ARM8
	// MIN AND MAX
	PLATFORM_MIN = PLATFORM_DARWIN
	PLATFORM_MAX = PLATFORM_BCM2838_ARM8
)

const (
	DISPLAY_FLAG_HDMI DisplayFlag = (1 << iota)
	DISPLAY_FLAG_DVI
	DISPLAY_FLAG_LCD
	DISPLAY_FLAG_SDTV
	DISPLAY_FLAG_NTSC
	DISPLAY_FLAG_PAL
	DISPLAY_FLAG_UNPLUGGED
	DISPLAY_FLAG_ATTACHED

	DISPLAY_FLAG_NONE DisplayFlag = 0
	DISPLAY_FLAG_MIN              = DISPLAY_FLAG_HDMI
	DISPLAY_FLAG_MAX              = DISPLAY_FLAG_ATTACHED
)

const (
	SPI_MODE_CPHA SPIMode = 0x01
	SPI_MODE_CPOL SPIMode = 0x02
	SPI_MODE_0    SPIMode = 0x00
	SPI_MODE_1    SPIMode = (0x00 | SPI_MODE_CPHA)
	SPI_MODE_2    SPIMode = (SPI_MODE_CPOL | 0x00)
	SPI_MODE_3    SPIMode = (SPI_MODE_CPOL | SPI_MODE_CPHA)
	SPI_MODE_NONE SPIMode = 0xFF
	SPI_MODE_MASK SPIMode = 0x03
)

const (
	// Invalid pin constant
	GPIO_PIN_NONE GPIOPin = 0xFF
)

const (
	GPIO_LOW GPIOState = iota
	GPIO_HIGH
)

const (
	// Set pin mode and/or function
	GPIO_INPUT GPIOMode = iota
	GPIO_OUTPUT
	GPIO_ALT5
	GPIO_ALT4
	GPIO_ALT0
	GPIO_ALT1
	GPIO_ALT2
	GPIO_ALT3
	GPIO_NONE
)

const (
	GPIO_PULL_OFF GPIOPull = iota
	GPIO_PULL_DOWN
	GPIO_PULL_UP
)

const (
	GPIO_EDGE_NONE GPIOEdge = iota
	GPIO_EDGE_RISING
	GPIO_EDGE_FALLING
	GPIO_EDGE_BOTH
)

const (
	LIRC_MODE_NONE     LIRCMode = 0x00000000
	LIRC_MODE_RAW      LIRCMode = 0x00000001
	LIRC_MODE_PULSE    LIRCMode = 0x00000002 // send only
	LIRC_MODE_MODE2    LIRCMode = 0x00000004 // rcv only
	LIRC_MODE_LIRCCODE LIRCMode = 0x00000010 // rcv only
)

const (
	LIRC_TYPE_SPACE     LIRCType = 0x00000000
	LIRC_TYPE_PULSE     LIRCType = 0x01000000
	LIRC_TYPE_FREQUENCY LIRCType = 0x02000000
	LIRC_TYPE_TIMEOUT   LIRCType = 0x03000000
)

////////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (p PlatformType) String() string {
	str := ""
	if p == 0 {
		return p.FlagString()
	}
	for v := PLATFORM_MIN; v <= PLATFORM_MAX; v <<= 1 {
		if p&v == v {
			str += "|" + v.FlagString()
		}
	}
	return strings.TrimPrefix(str, "|")
}

func (p PlatformType) FlagString() string {
	switch p {
	case PLATFORM_NONE:
		return "PLATFORM_NONE"
	case PLATFORM_DARWIN:
		return "PLATFORM_DARWIN"
	case PLATFORM_RPI:
		return "PLATFORM_RPI"
	case PLATFORM_LINUX:
		return "PLATFORM_LINUX"
	case PLATFORM_X86_32:
		return "PLATFORM_X86_32"
	case PLATFORM_X86_64:
		return "PLATFORM_X86_64"
	case PLATFORM_BCM2835_ARM6:
		return "PLATFORM_BCM2835_ARM6"
	case PLATFORM_BCM2836_ARM7:
		return "PLATFORM_BCM2836_ARM7"
	case PLATFORM_BCM2837_ARM8:
		return "PLATFORM_BCM2837_ARM8"
	case PLATFORM_BCM2838_ARM8:
		return "PLATFORM_BCM2838_ARM8"
	default:
		return "[?? Invalid PlatformType value]"
	}
}

func (m SPIMode) String() string {
	switch m & SPI_MODE_MASK {
	case SPI_MODE_NONE:
		return "SPI_MODE_NONE"
	case SPI_MODE_0:
		return "SPI_MODE_0"
	case SPI_MODE_1:
		return "SPI_MODE_1"
	case SPI_MODE_2:
		return "SPI_MODE_2"
	case SPI_MODE_3:
		return "SPI_MODE_3"
	default:
		return "[?? Invalid SPIMode " + fmt.Sprint(uint(m)) + "]"
	}
}

func (p GPIOPin) String() string {
	return fmt.Sprintf("GPIO%v", uint8(p))
}

func (s GPIOState) String() string {
	switch s {
	case GPIO_LOW:
		return "GPIO_LOW"
	case GPIO_HIGH:
		return "GPIO_HIGH"
	default:
		return "[??? Invalid GPIOState value]"
	}
}

func (m GPIOMode) String() string {
	switch m {
	case GPIO_INPUT:
		return "GPIO_INPUT"
	case GPIO_OUTPUT:
		return "GPIO_OUTPUT"
	case GPIO_ALT0:
		return "GPIO_ALT0"
	case GPIO_ALT1:
		return "GPIO_ALT1"
	case GPIO_ALT2:
		return "GPIO_ALT2"
	case GPIO_ALT3:
		return "GPIO_ALT3"
	case GPIO_ALT4:
		return "GPIO_ALT4"
	case GPIO_ALT5:
		return "GPIO_ALT5"
	case GPIO_NONE:
		return "GPIO_NONE"
	default:
		return "[??? Invalid GPIOMode value]"
	}
}

func (p GPIOPull) String() string {
	switch p {
	case GPIO_PULL_OFF:
		return "GPIO_PULL_OFF"
	case GPIO_PULL_DOWN:
		return "GPIO_PULL_DOWN"
	case GPIO_PULL_UP:
		return "GPIO_PULL_UP"
	default:
		return "[??? Invalid GPIOPull value]"
	}
}

func (e GPIOEdge) String() string {
	switch e {
	case GPIO_EDGE_NONE:
		return "GPIO_EDGE_NONE"
	case GPIO_EDGE_RISING:
		return "GPIO_EDGE_RISING"
	case GPIO_EDGE_FALLING:
		return "GPIO_EDGE_FALLING"
	case GPIO_EDGE_BOTH:
		return "GPIO_EDGE_BOTH"
	default:
		return "[??? Invalid GPIOEdge value]"
	}
}

func (m LIRCMode) String() string {
	switch m {
	case LIRC_MODE_NONE:
		return "LIRC_MODE_NONE"
	case LIRC_MODE_RAW:
		return "LIRC_MODE_RAW"
	case LIRC_MODE_PULSE:
		return "LIRC_MODE_PULSE"
	case LIRC_MODE_MODE2:
		return "LIRC_MODE_MODE2"
	case LIRC_MODE_LIRCCODE:
		return "LIRC_MODE_LIRCCODE"
	default:
		return "[?? Invalid LIRCMode value]"
	}
}

func (t LIRCType) String() string {
	switch t {
	case LIRC_TYPE_SPACE:
		return "LIRC_TYPE_SPACE"
	case LIRC_TYPE_PULSE:
		return "LIRC_TYPE_PULSE"
	case LIRC_TYPE_FREQUENCY:
		return "LIRC_TYPE_FREQUENCY"
	case LIRC_TYPE_TIMEOUT:
		return "LIRC_TYPE_TIMEOUT"
	default:
		return "[?? Invalid LIRCType value]"
	}
}

func (f DisplayFlag) String() string {
	if f == DISPLAY_FLAG_NONE {
		return f.FlagString()
	}
	str := ""
	for v := DISPLAY_FLAG_MIN; v <= DISPLAY_FLAG_MAX; v <<= 1 {
		if f&v == v {
			str += "|" + v.FlagString()
		}
	}
	return strings.TrimPrefix(str, "|")
}

func (f DisplayFlag) FlagString() string {
	switch f {
	case DISPLAY_FLAG_NONE:
		return "DISPLAY_FLAG_NONE"
	case DISPLAY_FLAG_HDMI:
		return "DISPLAY_FLAG_HDMI"
	case DISPLAY_FLAG_DVI:
		return "DISPLAY_FLAG_DVI"
	case DISPLAY_FLAG_LCD:
		return "DISPLAY_FLAG_LCD"
	case DISPLAY_FLAG_SDTV:
		return "DISPLAY_FLAG_SDTV"
	case DISPLAY_FLAG_NTSC:
		return "DISPLAY_FLAG_NTSC"
	case DISPLAY_FLAG_PAL:
		return "DISPLAY_FLAG_PAL"
	case DISPLAY_FLAG_UNPLUGGED:
		return "DISPLAY_FLAG_UNPLUGGED"
	case DISPLAY_FLAG_ATTACHED:
		return "DISPLAY_FLAG_ATTACHED"
	default:
		return "[?? Invalid DisplayFlag value]"
	}
}
