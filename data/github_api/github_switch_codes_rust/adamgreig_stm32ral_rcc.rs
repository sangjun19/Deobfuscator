// Repository: adamgreig/stm32ral
// File: src/stm32f4/stm32f427/rcc.rs

#![allow(non_snake_case, non_upper_case_globals)]
#![allow(non_camel_case_types)]
//! Reset and clock control

use crate::RWRegister;
#[cfg(not(feature = "nosync"))]
use core::marker::PhantomData;

/// clock control register
pub mod CR {

    /// PLLI2S clock ready flag
    pub mod PLLI2SRDY {
        /// Offset (27 bits)
        pub const offset: u32 = 27;
        /// Mask (1 bit: 1 << 27)
        pub const mask: u32 = 1 << offset;
        /// Read-only values
        pub mod R {

            /// 0b0: Clock not ready
            pub const NotReady: u32 = 0b0;

            /// 0b1: Clock ready
            pub const Ready: u32 = 0b1;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLI2S enable
    pub mod PLLI2SON {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Clock Off
            pub const Off: u32 = 0b0;

            /// 0b1: Clock On
            pub const On: u32 = 0b1;
        }
    }

    /// Main PLL (PLL) clock ready flag
    pub mod PLLRDY {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDY::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Main PLL (PLL) enable
    pub mod PLLON {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (1 bit: 1 << 24)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SON::RW;
    }

    /// Clock security system enable
    pub mod CSSON {
        /// Offset (19 bits)
        pub const offset: u32 = 19;
        /// Mask (1 bit: 1 << 19)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Clock security system disabled (clock detector OFF)
            pub const Off: u32 = 0b0;

            /// 0b1: Clock security system enable (clock detector ON if the HSE is ready, OFF if not)
            pub const On: u32 = 0b1;
        }
    }

    /// HSE clock bypass
    pub mod HSEBYP {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: HSE crystal oscillator not bypassed
            pub const NotBypassed: u32 = 0b0;

            /// 0b1: HSE crystal oscillator bypassed with external clock
            pub const Bypassed: u32 = 0b1;
        }
    }

    /// HSE clock ready flag
    pub mod HSERDY {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDY::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// HSE clock enable
    pub mod HSEON {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SON::RW;
    }

    /// Internal high-speed clock calibration
    pub mod HSICAL {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (8 bits: 0xff << 8)
        pub const mask: u32 = 0xff << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Internal high-speed clock trimming
    pub mod HSITRIM {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (5 bits: 0b11111 << 3)
        pub const mask: u32 = 0b11111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Internal high-speed clock ready flag
    pub mod HSIRDY {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDY::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Internal high-speed clock enable
    pub mod HSION {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SON::RW;
    }

    /// PLLSAI clock ready flag
    pub mod PLLSAIRDY {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDY::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLSAI enable
    pub mod PLLSAION {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SON::RW;
    }
}

/// PLL configuration register
pub mod PLLCFGR {

    /// Main PLL(PLL) and audio PLL (PLLI2S) entry clock source
    pub mod PLLSRC {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: HSI clock selected as PLL and PLLI2S clock entry
            pub const HSI: u32 = 0b0;

            /// 0b1: HSE oscillator clock selected as PLL and PLLI2S clock entry
            pub const HSE: u32 = 0b1;
        }
    }

    /// Division factor for the main PLL (PLL) and audio PLL (PLLI2S) input clock
    pub mod PLLM {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (6 bits: 0x3f << 0)
        pub const mask: u32 = 0x3f << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Main PLL (PLL) multiplication factor for VCO
    pub mod PLLN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (9 bits: 0x1ff << 6)
        pub const mask: u32 = 0x1ff << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Main PLL (PLL) division factor for main system clock
    pub mod PLLP {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (2 bits: 0b11 << 16)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: PLLP=2
            pub const Div2: u32 = 0b00;

            /// 0b01: PLLP=4
            pub const Div4: u32 = 0b01;

            /// 0b10: PLLP=6
            pub const Div6: u32 = 0b10;

            /// 0b11: PLLP=8
            pub const Div8: u32 = 0b11;
        }
    }

    /// Main PLL (PLL) division factor for USB OTG FS, SDIO and random number generator clocks
    pub mod PLLQ {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (4 bits: 0b1111 << 24)
        pub const mask: u32 = 0b1111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }
}

/// clock configuration register
pub mod CFGR {

    /// Microcontroller clock output 2
    pub mod MCO2 {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (2 bits: 0b11 << 30)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: System clock (SYSCLK) selected
            pub const SYSCLK: u32 = 0b00;

            /// 0b01: PLLI2S clock selected
            pub const PLLI2S: u32 = 0b01;

            /// 0b10: HSE oscillator clock selected
            pub const HSE: u32 = 0b10;

            /// 0b11: PLL clock selected
            pub const PLL: u32 = 0b11;
        }
    }

    /// MCO2 prescaler
    pub mod MCO2PRE {
        /// Offset (27 bits)
        pub const offset: u32 = 27;
        /// Mask (3 bits: 0b111 << 27)
        pub const mask: u32 = 0b111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b000: No division
            pub const Div1: u32 = 0b000;

            /// 0b100: Division by 2
            pub const Div2: u32 = 0b100;

            /// 0b101: Division by 3
            pub const Div3: u32 = 0b101;

            /// 0b110: Division by 4
            pub const Div4: u32 = 0b110;

            /// 0b111: Division by 5
            pub const Div5: u32 = 0b111;
        }
    }

    /// MCO1 prescaler
    pub mod MCO1PRE {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (3 bits: 0b111 << 24)
        pub const mask: u32 = 0b111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::MCO2PRE::RW;
    }

    /// I2S clock selection
    pub mod I2SSRC {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: PLLI2S clock used as I2S clock source
            pub const PLLI2S: u32 = 0b0;

            /// 0b1: External clock mapped on the I2S_CKIN pin used as I2S clock source
            pub const CKIN: u32 = 0b1;
        }
    }

    /// Microcontroller clock output 1
    pub mod MCO1 {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (2 bits: 0b11 << 21)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: HSI clock selected
            pub const HSI: u32 = 0b00;

            /// 0b01: LSE oscillator selected
            pub const LSE: u32 = 0b01;

            /// 0b10: HSE oscillator clock selected
            pub const HSE: u32 = 0b10;

            /// 0b11: PLL clock selected
            pub const PLL: u32 = 0b11;
        }
    }

    /// HSE division factor for RTC clock
    pub mod RTCPRE {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (5 bits: 0b11111 << 16)
        pub const mask: u32 = 0b11111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// APB high-speed prescaler (APB2)
    pub mod PPRE2 {
        /// Offset (13 bits)
        pub const offset: u32 = 13;
        /// Mask (3 bits: 0b111 << 13)
        pub const mask: u32 = 0b111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b000: HCLK not divided
            pub const Div1: u32 = 0b000;

            /// 0b100: HCLK divided by 2
            pub const Div2: u32 = 0b100;

            /// 0b101: HCLK divided by 4
            pub const Div4: u32 = 0b101;

            /// 0b110: HCLK divided by 8
            pub const Div8: u32 = 0b110;

            /// 0b111: HCLK divided by 16
            pub const Div16: u32 = 0b111;
        }
    }

    /// APB Low speed prescaler (APB1)
    pub mod PPRE1 {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (3 bits: 0b111 << 10)
        pub const mask: u32 = 0b111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PPRE2::RW;
    }

    /// AHB prescaler
    pub mod HPRE {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (4 bits: 0b1111 << 4)
        pub const mask: u32 = 0b1111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0000: SYSCLK not divided
            pub const Div1: u32 = 0b0000;

            /// 0b1000: SYSCLK divided by 2
            pub const Div2: u32 = 0b1000;

            /// 0b1001: SYSCLK divided by 4
            pub const Div4: u32 = 0b1001;

            /// 0b1010: SYSCLK divided by 8
            pub const Div8: u32 = 0b1010;

            /// 0b1011: SYSCLK divided by 16
            pub const Div16: u32 = 0b1011;

            /// 0b1100: SYSCLK divided by 64
            pub const Div64: u32 = 0b1100;

            /// 0b1101: SYSCLK divided by 128
            pub const Div128: u32 = 0b1101;

            /// 0b1110: SYSCLK divided by 256
            pub const Div256: u32 = 0b1110;

            /// 0b1111: SYSCLK divided by 512
            pub const Div512: u32 = 0b1111;
        }
    }

    /// System clock switch
    pub mod SW {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (2 bits: 0b11 << 0)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: HSI selected as system clock
            pub const HSI: u32 = 0b00;

            /// 0b01: HSE selected as system clock
            pub const HSE: u32 = 0b01;

            /// 0b10: PLL selected as system clock
            pub const PLL: u32 = 0b10;
        }
    }

    /// System clock switch status
    pub mod SWS {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (2 bits: 0b11 << 2)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values
        pub mod R {

            /// 0b00: HSI oscillator used as system clock
            pub const HSI: u32 = 0b00;

            /// 0b01: HSE oscillator used as system clock
            pub const HSE: u32 = 0b01;

            /// 0b10: PLL used as system clock
            pub const PLL: u32 = 0b10;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }
}

/// clock interrupt register
pub mod CIR {

    /// Clock security system interrupt clear
    pub mod CSSC {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values
        pub mod W {

            /// 0b1: Clear CSSF flag
            pub const Clear: u32 = 0b1;
        }
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLI2S ready interrupt clear
    pub mod PLLI2SRDYC {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values
        pub mod W {

            /// 0b1: Clear interrupt flag
            pub const Clear: u32 = 0b1;
        }
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Main PLL(PLL) ready interrupt clear
    pub mod PLLRDYC {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        pub use super::PLLI2SRDYC::W;
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// HSE ready interrupt clear
    pub mod HSERDYC {
        /// Offset (19 bits)
        pub const offset: u32 = 19;
        /// Mask (1 bit: 1 << 19)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        pub use super::PLLI2SRDYC::W;
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// HSI ready interrupt clear
    pub mod HSIRDYC {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        pub use super::PLLI2SRDYC::W;
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// LSE ready interrupt clear
    pub mod LSERDYC {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        pub use super::PLLI2SRDYC::W;
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// LSI ready interrupt clear
    pub mod LSIRDYC {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        pub use super::PLLI2SRDYC::W;
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLI2S ready interrupt enable
    pub mod PLLI2SRDYIE {
        /// Offset (13 bits)
        pub const offset: u32 = 13;
        /// Mask (1 bit: 1 << 13)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Interrupt disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: Interrupt enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// Main PLL (PLL) ready interrupt enable
    pub mod PLLRDYIE {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SRDYIE::RW;
    }

    /// HSE ready interrupt enable
    pub mod HSERDYIE {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SRDYIE::RW;
    }

    /// HSI ready interrupt enable
    pub mod HSIRDYIE {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (1 bit: 1 << 10)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SRDYIE::RW;
    }

    /// LSE ready interrupt enable
    pub mod LSERDYIE {
        /// Offset (9 bits)
        pub const offset: u32 = 9;
        /// Mask (1 bit: 1 << 9)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SRDYIE::RW;
    }

    /// LSI ready interrupt enable
    pub mod LSIRDYIE {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::PLLI2SRDYIE::RW;
    }

    /// Clock security system interrupt flag
    pub mod CSSF {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values
        pub mod R {

            /// 0b0: No clock security interrupt caused by HSE clock failure
            pub const NotInterrupted: u32 = 0b0;

            /// 0b1: Clock security interrupt caused by HSE clock failure
            pub const Interrupted: u32 = 0b1;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLI2S ready interrupt flag
    pub mod PLLI2SRDYF {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values
        pub mod R {

            /// 0b0: No clock ready interrupt
            pub const NotInterrupted: u32 = 0b0;

            /// 0b1: Clock ready interrupt
            pub const Interrupted: u32 = 0b1;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Main PLL (PLL) ready interrupt flag
    pub mod PLLRDYF {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDYF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// HSE ready interrupt flag
    pub mod HSERDYF {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDYF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// HSI ready interrupt flag
    pub mod HSIRDYF {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDYF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// LSE ready interrupt flag
    pub mod LSERDYF {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDYF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// LSI ready interrupt flag
    pub mod LSIRDYF {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        pub use super::PLLI2SRDYF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }
}

/// AHB1 peripheral reset register
pub mod AHB1RSTR {

    /// USB OTG HS module reset
    pub mod OTGHSRST {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b1: Reset the selected module
            pub const Reset: u32 = 0b1;
        }
    }

    /// Ethernet MAC reset
    pub mod ETHMACRST {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// DMA2 reset
    pub mod DMA2RST {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// DMA2 reset
    pub mod DMA1RST {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// CRC reset
    pub mod CRCRST {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port I reset
    pub mod GPIOIRST {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port H reset
    pub mod GPIOHRST {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port G reset
    pub mod GPIOGRST {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port F reset
    pub mod GPIOFRST {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port E reset
    pub mod GPIOERST {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port D reset
    pub mod GPIODRST {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port C reset
    pub mod GPIOCRST {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port B reset
    pub mod GPIOBRST {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port A reset
    pub mod GPIOARST {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// DMA2D reset
    pub mod DMA2DRST {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port J reset
    pub mod GPIOJRST {
        /// Offset (9 bits)
        pub const offset: u32 = 9;
        /// Mask (1 bit: 1 << 9)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }

    /// IO port K reset
    pub mod GPIOKRST {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (1 bit: 1 << 10)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSRST::RW;
    }
}

/// AHB2 peripheral reset register
pub mod AHB2RSTR {

    /// USB OTG FS module reset
    pub mod OTGFSRST {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b1: Reset the selected module
            pub const Reset: u32 = 0b1;
        }
    }

    /// Random number generator module reset
    pub mod RNGRST {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSRST::RW;
    }

    /// Hash module reset
    pub mod HSAHRST {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSRST::RW;
    }

    /// Cryptographic module reset
    pub mod CRYPRST {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSRST::RW;
    }

    /// Camera interface reset
    pub mod DCMIRST {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSRST::RW;
    }
}

/// AHB3 peripheral reset register
pub mod AHB3RSTR {

    /// Flexible static memory controller module reset
    pub mod FMCRST {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b1: Reset the selected module
            pub const Reset: u32 = 0b1;
        }
    }
}

/// APB1 peripheral reset register
pub mod APB1RSTR {

    /// TIM2 reset
    pub mod TIM2RST {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b1: Reset the selected module
            pub const Reset: u32 = 0b1;
        }
    }

    /// TIM3 reset
    pub mod TIM3RST {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM4 reset
    pub mod TIM4RST {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM5 reset
    pub mod TIM5RST {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM6 reset
    pub mod TIM6RST {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM7 reset
    pub mod TIM7RST {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM12 reset
    pub mod TIM12RST {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM13 reset
    pub mod TIM13RST {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// TIM14 reset
    pub mod TIM14RST {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// Window watchdog reset
    pub mod WWDGRST {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// SPI 2 reset
    pub mod SPI2RST {
        /// Offset (14 bits)
        pub const offset: u32 = 14;
        /// Mask (1 bit: 1 << 14)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// SPI 3 reset
    pub mod SPI3RST {
        /// Offset (15 bits)
        pub const offset: u32 = 15;
        /// Mask (1 bit: 1 << 15)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// USART 2 reset
    pub mod USART2RST {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// USART 3 reset
    pub mod USART3RST {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// USART 4 reset
    pub mod UART4RST {
        /// Offset (19 bits)
        pub const offset: u32 = 19;
        /// Mask (1 bit: 1 << 19)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// USART 5 reset
    pub mod UART5RST {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// I2C 1 reset
    pub mod I2C1RST {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// I2C 2 reset
    pub mod I2C2RST {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// I2C3 reset
    pub mod I2C3RST {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// CAN1 reset
    pub mod CAN1RST {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// CAN2 reset
    pub mod CAN2RST {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// Power interface reset
    pub mod PWRRST {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// DAC reset
    pub mod DACRST {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// UART7 reset
    pub mod UART7RST {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }

    /// UART8 reset
    pub mod UART8RST {
        /// Offset (31 bits)
        pub const offset: u32 = 31;
        /// Mask (1 bit: 1 << 31)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2RST::RW;
    }
}

/// APB2 peripheral reset register
pub mod APB2RSTR {

    /// TIM1 reset
    pub mod TIM1RST {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b1: Reset the selected module
            pub const Reset: u32 = 0b1;
        }
    }

    /// TIM8 reset
    pub mod TIM8RST {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// USART1 reset
    pub mod USART1RST {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// USART6 reset
    pub mod USART6RST {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// ADC interface reset (common to all ADCs)
    pub mod ADCRST {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// SDIO reset
    pub mod SDIORST {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// SPI 1 reset
    pub mod SPI1RST {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// SPI4 reset
    pub mod SPI4RST {
        /// Offset (13 bits)
        pub const offset: u32 = 13;
        /// Mask (1 bit: 1 << 13)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// System configuration controller reset
    pub mod SYSCFGRST {
        /// Offset (14 bits)
        pub const offset: u32 = 14;
        /// Mask (1 bit: 1 << 14)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// TIM9 reset
    pub mod TIM9RST {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// TIM10 reset
    pub mod TIM10RST {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// TIM11 reset
    pub mod TIM11RST {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// SPI5 reset
    pub mod SPI5RST {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// SPI6 reset
    pub mod SPI6RST {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// SAI1 reset
    pub mod SAI1RST {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }

    /// LTDC reset
    pub mod LTDCRST {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1RST::RW;
    }
}

/// AHB1 peripheral clock register
pub mod AHB1ENR {

    /// USB OTG HSULPI clock enable
    pub mod OTGHSULPIEN {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: The selected clock is disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: The selected clock is enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// USB OTG HS clock enable
    pub mod OTGHSEN {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// Ethernet PTP clock enable
    pub mod ETHMACPTPEN {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// Ethernet Reception clock enable
    pub mod ETHMACRXEN {
        /// Offset (27 bits)
        pub const offset: u32 = 27;
        /// Mask (1 bit: 1 << 27)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// Ethernet Transmission clock enable
    pub mod ETHMACTXEN {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// Ethernet MAC clock enable
    pub mod ETHMACEN {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// DMA2 clock enable
    pub mod DMA2EN {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// DMA1 clock enable
    pub mod DMA1EN {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// Backup SRAM interface clock enable
    pub mod BKPSRAMEN {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// CRC clock enable
    pub mod CRCEN {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port I clock enable
    pub mod GPIOIEN {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port H clock enable
    pub mod GPIOHEN {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port G clock enable
    pub mod GPIOGEN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port F clock enable
    pub mod GPIOFEN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port E clock enable
    pub mod GPIOEEN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port D clock enable
    pub mod GPIODEN {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port C clock enable
    pub mod GPIOCEN {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port B clock enable
    pub mod GPIOBEN {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port A clock enable
    pub mod GPIOAEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// DMA2D clock enable
    pub mod DMA2DEN {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port J clock enable
    pub mod GPIOJEN {
        /// Offset (9 bits)
        pub const offset: u32 = 9;
        /// Mask (1 bit: 1 << 9)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }

    /// IO port K clock enable
    pub mod GPIOKEN {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (1 bit: 1 << 10)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGHSULPIEN::RW;
    }
}

/// AHB2 peripheral clock enable register
pub mod AHB2ENR {

    /// USB OTG FS clock enable
    pub mod OTGFSEN {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: The selected clock is disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: The selected clock is enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// Random number generator clock enable
    pub mod RNGEN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSEN::RW;
    }

    /// Hash modules clock enable
    pub mod HASHEN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSEN::RW;
    }

    /// Cryptographic modules clock enable
    pub mod CRYPEN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSEN::RW;
    }

    /// Camera interface enable
    pub mod DCMIEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSEN::RW;
    }
}

/// AHB3 peripheral clock enable register
pub mod AHB3ENR {

    /// Flexible static memory controller module clock enable
    pub mod FMCEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: The selected clock is disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: The selected clock is enabled
            pub const Enabled: u32 = 0b1;
        }
    }
}

/// APB1 peripheral clock enable register
pub mod APB1ENR {

    /// TIM2 clock enable
    pub mod TIM2EN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: The selected clock is disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: The selected clock is enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// TIM3 clock enable
    pub mod TIM3EN {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM4 clock enable
    pub mod TIM4EN {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM5 clock enable
    pub mod TIM5EN {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM6 clock enable
    pub mod TIM6EN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM7 clock enable
    pub mod TIM7EN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM12 clock enable
    pub mod TIM12EN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM13 clock enable
    pub mod TIM13EN {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// TIM14 clock enable
    pub mod TIM14EN {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// Window watchdog clock enable
    pub mod WWDGEN {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// SPI2 clock enable
    pub mod SPI2EN {
        /// Offset (14 bits)
        pub const offset: u32 = 14;
        /// Mask (1 bit: 1 << 14)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// SPI3 clock enable
    pub mod SPI3EN {
        /// Offset (15 bits)
        pub const offset: u32 = 15;
        /// Mask (1 bit: 1 << 15)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// USART 2 clock enable
    pub mod USART2EN {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// USART3 clock enable
    pub mod USART3EN {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// UART4 clock enable
    pub mod UART4EN {
        /// Offset (19 bits)
        pub const offset: u32 = 19;
        /// Mask (1 bit: 1 << 19)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// UART5 clock enable
    pub mod UART5EN {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// I2C1 clock enable
    pub mod I2C1EN {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// I2C2 clock enable
    pub mod I2C2EN {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// I2C3 clock enable
    pub mod I2C3EN {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// CAN 1 clock enable
    pub mod CAN1EN {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// CAN 2 clock enable
    pub mod CAN2EN {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// Power interface clock enable
    pub mod PWREN {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// DAC interface clock enable
    pub mod DACEN {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// UART7 clock enable
    pub mod UART7EN {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }

    /// UART8 clock enable
    pub mod UART8EN {
        /// Offset (31 bits)
        pub const offset: u32 = 31;
        /// Mask (1 bit: 1 << 31)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2EN::RW;
    }
}

/// APB2 peripheral clock enable register
pub mod APB2ENR {

    /// TIM1 clock enable
    pub mod TIM1EN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: The selected clock is disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: The selected clock is enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// TIM8 clock enable
    pub mod TIM8EN {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// USART1 clock enable
    pub mod USART1EN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// USART6 clock enable
    pub mod USART6EN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// ADC1 clock enable
    pub mod ADC1EN {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// ADC2 clock enable
    pub mod ADC2EN {
        /// Offset (9 bits)
        pub const offset: u32 = 9;
        /// Mask (1 bit: 1 << 9)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// ADC3 clock enable
    pub mod ADC3EN {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (1 bit: 1 << 10)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// SDIO clock enable
    pub mod SDIOEN {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// SPI1 clock enable
    pub mod SPI1EN {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// SPI4 clock enable
    pub mod SPI4EN {
        /// Offset (13 bits)
        pub const offset: u32 = 13;
        /// Mask (1 bit: 1 << 13)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// System configuration controller clock enable
    pub mod SYSCFGEN {
        /// Offset (14 bits)
        pub const offset: u32 = 14;
        /// Mask (1 bit: 1 << 14)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// TIM9 clock enable
    pub mod TIM9EN {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// TIM10 clock enable
    pub mod TIM10EN {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// TIM11 clock enable
    pub mod TIM11EN {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// SPI5 clock enable
    pub mod SPI5EN {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// SPI6 clock enable
    pub mod SPI6EN {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// SAI1 clock enable
    pub mod SAI1EN {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }

    /// LTDC clock enable
    pub mod LTDCEN {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1EN::RW;
    }
}

/// AHB1 peripheral clock enable in low power mode register
pub mod AHB1LPENR {

    /// IO port A clock enable during sleep mode
    pub mod GPIOALPEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Selected module is disabled during Sleep mode
            pub const DisabledInSleep: u32 = 0b0;

            /// 0b1: Selected module is enabled during Sleep mode
            pub const EnabledInSleep: u32 = 0b1;
        }
    }

    /// IO port B clock enable during Sleep mode
    pub mod GPIOBLPEN {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port C clock enable during Sleep mode
    pub mod GPIOCLPEN {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port D clock enable during Sleep mode
    pub mod GPIODLPEN {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port E clock enable during Sleep mode
    pub mod GPIOELPEN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port F clock enable during Sleep mode
    pub mod GPIOFLPEN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port G clock enable during Sleep mode
    pub mod GPIOGLPEN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port H clock enable during Sleep mode
    pub mod GPIOHLPEN {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port I clock enable during Sleep mode
    pub mod GPIOILPEN {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// CRC clock enable during Sleep mode
    pub mod CRCLPEN {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// Flash interface clock enable during Sleep mode
    pub mod FLITFLPEN {
        /// Offset (15 bits)
        pub const offset: u32 = 15;
        /// Mask (1 bit: 1 << 15)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// SRAM 1interface clock enable during Sleep mode
    pub mod SRAM1LPEN {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// SRAM 2 interface clock enable during Sleep mode
    pub mod SRAM2LPEN {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// Backup SRAM interface clock enable during Sleep mode
    pub mod BKPSRAMLPEN {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// SRAM 3 interface clock enable during Sleep mode
    pub mod SRAM3LPEN {
        /// Offset (19 bits)
        pub const offset: u32 = 19;
        /// Mask (1 bit: 1 << 19)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// DMA1 clock enable during Sleep mode
    pub mod DMA1LPEN {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// DMA2 clock enable during Sleep mode
    pub mod DMA2LPEN {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// Ethernet MAC clock enable during Sleep mode
    pub mod ETHMACLPEN {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// Ethernet transmission clock enable during Sleep mode
    pub mod ETHMACTXLPEN {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// Ethernet reception clock enable during Sleep mode
    pub mod ETHMACRXLPEN {
        /// Offset (27 bits)
        pub const offset: u32 = 27;
        /// Mask (1 bit: 1 << 27)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// Ethernet PTP clock enable during Sleep mode
    pub mod ETHMACPTPLPEN {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// USB OTG HS clock enable during Sleep mode
    pub mod OTGHSLPEN {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// USB OTG HS ULPI clock enable during Sleep mode
    pub mod OTGHSULPILPEN {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// DMA2D clock enable during Sleep mode
    pub mod DMA2DLPEN {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port J clock enable during Sleep mode
    pub mod GPIOJLPEN {
        /// Offset (9 bits)
        pub const offset: u32 = 9;
        /// Mask (1 bit: 1 << 9)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }

    /// IO port K clock enable during Sleep mode
    pub mod GPIOKLPEN {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (1 bit: 1 << 10)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::GPIOALPEN::RW;
    }
}

/// AHB2 peripheral clock enable in low power mode register
pub mod AHB2LPENR {

    /// USB OTG FS clock enable during Sleep mode
    pub mod OTGFSLPEN {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Selected module is disabled during Sleep mode
            pub const DisabledInSleep: u32 = 0b0;

            /// 0b1: Selected module is enabled during Sleep mode
            pub const EnabledInSleep: u32 = 0b1;
        }
    }

    /// Random number generator clock enable during Sleep mode
    pub mod RNGLPEN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSLPEN::RW;
    }

    /// Hash modules clock enable during Sleep mode
    pub mod HASHLPEN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSLPEN::RW;
    }

    /// Cryptography modules clock enable during Sleep mode
    pub mod CRYPLPEN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSLPEN::RW;
    }

    /// Camera interface enable during Sleep mode
    pub mod DCMILPEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::OTGFSLPEN::RW;
    }
}

/// AHB3 peripheral clock enable in low power mode register
pub mod AHB3LPENR {

    /// Flexible static memory controller module clock enable during Sleep mode
    pub mod FMCLPEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Selected module is disabled during Sleep mode
            pub const DisabledInSleep: u32 = 0b0;

            /// 0b1: Selected module is enabled during Sleep mode
            pub const EnabledInSleep: u32 = 0b1;
        }
    }
}

/// APB1 peripheral clock enable in low power mode register
pub mod APB1LPENR {

    /// TIM2 clock enable during Sleep mode
    pub mod TIM2LPEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Selected module is disabled during Sleep mode
            pub const DisabledInSleep: u32 = 0b0;

            /// 0b1: Selected module is enabled during Sleep mode
            pub const EnabledInSleep: u32 = 0b1;
        }
    }

    /// TIM3 clock enable during Sleep mode
    pub mod TIM3LPEN {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM4 clock enable during Sleep mode
    pub mod TIM4LPEN {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM5 clock enable during Sleep mode
    pub mod TIM5LPEN {
        /// Offset (3 bits)
        pub const offset: u32 = 3;
        /// Mask (1 bit: 1 << 3)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM6 clock enable during Sleep mode
    pub mod TIM6LPEN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM7 clock enable during Sleep mode
    pub mod TIM7LPEN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM12 clock enable during Sleep mode
    pub mod TIM12LPEN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (1 bit: 1 << 6)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM13 clock enable during Sleep mode
    pub mod TIM13LPEN {
        /// Offset (7 bits)
        pub const offset: u32 = 7;
        /// Mask (1 bit: 1 << 7)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// TIM14 clock enable during Sleep mode
    pub mod TIM14LPEN {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// Window watchdog clock enable during Sleep mode
    pub mod WWDGLPEN {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// SPI2 clock enable during Sleep mode
    pub mod SPI2LPEN {
        /// Offset (14 bits)
        pub const offset: u32 = 14;
        /// Mask (1 bit: 1 << 14)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// SPI3 clock enable during Sleep mode
    pub mod SPI3LPEN {
        /// Offset (15 bits)
        pub const offset: u32 = 15;
        /// Mask (1 bit: 1 << 15)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// USART2 clock enable during Sleep mode
    pub mod USART2LPEN {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// USART3 clock enable during Sleep mode
    pub mod USART3LPEN {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// UART4 clock enable during Sleep mode
    pub mod UART4LPEN {
        /// Offset (19 bits)
        pub const offset: u32 = 19;
        /// Mask (1 bit: 1 << 19)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// UART5 clock enable during Sleep mode
    pub mod UART5LPEN {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// I2C1 clock enable during Sleep mode
    pub mod I2C1LPEN {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// I2C2 clock enable during Sleep mode
    pub mod I2C2LPEN {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// I2C3 clock enable during Sleep mode
    pub mod I2C3LPEN {
        /// Offset (23 bits)
        pub const offset: u32 = 23;
        /// Mask (1 bit: 1 << 23)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// CAN 1 clock enable during Sleep mode
    pub mod CAN1LPEN {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// CAN 2 clock enable during Sleep mode
    pub mod CAN2LPEN {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// Power interface clock enable during Sleep mode
    pub mod PWRLPEN {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// DAC interface clock enable during Sleep mode
    pub mod DACLPEN {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// UART7 clock enable during Sleep mode
    pub mod UART7LPEN {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }

    /// UART8 clock enable during Sleep mode
    pub mod UART8LPEN {
        /// Offset (31 bits)
        pub const offset: u32 = 31;
        /// Mask (1 bit: 1 << 31)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM2LPEN::RW;
    }
}

/// APB2 peripheral clock enabled in low power mode register
pub mod APB2LPENR {

    /// TIM1 clock enable during Sleep mode
    pub mod TIM1LPEN {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Selected module is disabled during Sleep mode
            pub const DisabledInSleep: u32 = 0b0;

            /// 0b1: Selected module is enabled during Sleep mode
            pub const EnabledInSleep: u32 = 0b1;
        }
    }

    /// TIM8 clock enable during Sleep mode
    pub mod TIM8LPEN {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// USART1 clock enable during Sleep mode
    pub mod USART1LPEN {
        /// Offset (4 bits)
        pub const offset: u32 = 4;
        /// Mask (1 bit: 1 << 4)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// USART6 clock enable during Sleep mode
    pub mod USART6LPEN {
        /// Offset (5 bits)
        pub const offset: u32 = 5;
        /// Mask (1 bit: 1 << 5)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// ADC1 clock enable during Sleep mode
    pub mod ADC1LPEN {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (1 bit: 1 << 8)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// ADC2 clock enable during Sleep mode
    pub mod ADC2LPEN {
        /// Offset (9 bits)
        pub const offset: u32 = 9;
        /// Mask (1 bit: 1 << 9)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// ADC 3 clock enable during Sleep mode
    pub mod ADC3LPEN {
        /// Offset (10 bits)
        pub const offset: u32 = 10;
        /// Mask (1 bit: 1 << 10)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// SDIO clock enable during Sleep mode
    pub mod SDIOLPEN {
        /// Offset (11 bits)
        pub const offset: u32 = 11;
        /// Mask (1 bit: 1 << 11)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// SPI 1 clock enable during Sleep mode
    pub mod SPI1LPEN {
        /// Offset (12 bits)
        pub const offset: u32 = 12;
        /// Mask (1 bit: 1 << 12)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// SPI 4 clock enable during Sleep mode
    pub mod SPI4LPEN {
        /// Offset (13 bits)
        pub const offset: u32 = 13;
        /// Mask (1 bit: 1 << 13)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// System configuration controller clock enable during Sleep mode
    pub mod SYSCFGLPEN {
        /// Offset (14 bits)
        pub const offset: u32 = 14;
        /// Mask (1 bit: 1 << 14)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// TIM9 clock enable during sleep mode
    pub mod TIM9LPEN {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// TIM10 clock enable during Sleep mode
    pub mod TIM10LPEN {
        /// Offset (17 bits)
        pub const offset: u32 = 17;
        /// Mask (1 bit: 1 << 17)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// TIM11 clock enable during Sleep mode
    pub mod TIM11LPEN {
        /// Offset (18 bits)
        pub const offset: u32 = 18;
        /// Mask (1 bit: 1 << 18)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// SPI 5 clock enable during Sleep mode
    pub mod SPI5LPEN {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (1 bit: 1 << 20)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// SPI 6 clock enable during Sleep mode
    pub mod SPI6LPEN {
        /// Offset (21 bits)
        pub const offset: u32 = 21;
        /// Mask (1 bit: 1 << 21)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// SAI1 clock enable during Sleep mode
    pub mod SAI1LPEN {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (1 bit: 1 << 22)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }

    /// LTDC clock enable during Sleep mode
    pub mod LTDCLPEN {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        pub use super::TIM1LPEN::RW;
    }
}

/// Backup domain control register
pub mod BDCR {

    /// Backup domain software reset
    pub mod BDRST {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (1 bit: 1 << 16)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Reset not activated
            pub const Disabled: u32 = 0b0;

            /// 0b1: Reset the entire RTC domain
            pub const Enabled: u32 = 0b1;
        }
    }

    /// RTC clock enable
    pub mod RTCEN {
        /// Offset (15 bits)
        pub const offset: u32 = 15;
        /// Mask (1 bit: 1 << 15)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: RTC clock disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: RTC clock enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// External low-speed oscillator bypass
    pub mod LSEBYP {
        /// Offset (2 bits)
        pub const offset: u32 = 2;
        /// Mask (1 bit: 1 << 2)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: LSE crystal oscillator not bypassed
            pub const NotBypassed: u32 = 0b0;

            /// 0b1: LSE crystal oscillator bypassed with external clock
            pub const Bypassed: u32 = 0b1;
        }
    }

    /// External low-speed oscillator ready
    pub mod LSERDY {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values
        pub mod R {

            /// 0b0: LSE oscillator not ready
            pub const NotReady: u32 = 0b0;

            /// 0b1: LSE oscillator ready
            pub const Ready: u32 = 0b1;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// External low-speed oscillator enable
    pub mod LSEON {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: LSE oscillator Off
            pub const Off: u32 = 0b0;

            /// 0b1: LSE oscillator On
            pub const On: u32 = 0b1;
        }
    }

    /// RTC clock source selection
    pub mod RTCSEL {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (2 bits: 0b11 << 8)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: No clock
            pub const NoClock: u32 = 0b00;

            /// 0b01: LSE oscillator clock used as RTC clock
            pub const LSE: u32 = 0b01;

            /// 0b10: LSI oscillator clock used as RTC clock
            pub const LSI: u32 = 0b10;

            /// 0b11: HSE oscillator clock divided by a prescaler used as RTC clock
            pub const HSE: u32 = 0b11;
        }
    }
}

/// clock control & status register
pub mod CSR {

    /// Low-power reset flag
    pub mod LPWRRSTF {
        /// Offset (31 bits)
        pub const offset: u32 = 31;
        /// Mask (1 bit: 1 << 31)
        pub const mask: u32 = 1 << offset;
        /// Read-only values
        pub mod R {

            /// 0b0: No reset has occured
            pub const NoReset: u32 = 0b0;

            /// 0b1: A reset has occured
            pub const Reset: u32 = 0b1;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Window watchdog reset flag
    pub mod WWDGRSTF {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        pub use super::LPWRRSTF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Independent watchdog reset flag
    pub mod WDGRSTF {
        /// Offset (29 bits)
        pub const offset: u32 = 29;
        /// Mask (1 bit: 1 << 29)
        pub const mask: u32 = 1 << offset;
        pub use super::LPWRRSTF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Software reset flag
    pub mod SFTRSTF {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (1 bit: 1 << 28)
        pub const mask: u32 = 1 << offset;
        pub use super::LPWRRSTF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// POR/PDR reset flag
    pub mod PORRSTF {
        /// Offset (27 bits)
        pub const offset: u32 = 27;
        /// Mask (1 bit: 1 << 27)
        pub const mask: u32 = 1 << offset;
        pub use super::LPWRRSTF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PIN reset flag
    pub mod PADRSTF {
        /// Offset (26 bits)
        pub const offset: u32 = 26;
        /// Mask (1 bit: 1 << 26)
        pub const mask: u32 = 1 << offset;
        pub use super::LPWRRSTF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// BOR reset flag
    pub mod BORRSTF {
        /// Offset (25 bits)
        pub const offset: u32 = 25;
        /// Mask (1 bit: 1 << 25)
        pub const mask: u32 = 1 << offset;
        pub use super::LPWRRSTF::R;
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Remove reset flag
    pub mod RMVF {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (1 bit: 1 << 24)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values
        pub mod W {

            /// 0b1: Clears the reset flag
            pub const Clear: u32 = 0b1;
        }
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Internal low-speed oscillator ready
    pub mod LSIRDY {
        /// Offset (1 bits)
        pub const offset: u32 = 1;
        /// Mask (1 bit: 1 << 1)
        pub const mask: u32 = 1 << offset;
        /// Read-only values
        pub mod R {

            /// 0b0: LSI oscillator not ready
            pub const NotReady: u32 = 0b0;

            /// 0b1: LSI oscillator ready
            pub const Ready: u32 = 0b1;
        }
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Internal low-speed oscillator enable
    pub mod LSION {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (1 bit: 1 << 0)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: LSI oscillator Off
            pub const Off: u32 = 0b0;

            /// 0b1: LSI oscillator On
            pub const On: u32 = 0b1;
        }
    }
}

/// spread spectrum clock generation register
pub mod SSCGR {

    /// Spread spectrum modulation enable
    pub mod SSCGEN {
        /// Offset (31 bits)
        pub const offset: u32 = 31;
        /// Mask (1 bit: 1 << 31)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Spread spectrum modulation disabled
            pub const Disabled: u32 = 0b0;

            /// 0b1: Spread spectrum modulation enabled
            pub const Enabled: u32 = 0b1;
        }
    }

    /// Spread Select
    pub mod SPREADSEL {
        /// Offset (30 bits)
        pub const offset: u32 = 30;
        /// Mask (1 bit: 1 << 30)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: Center spread
            pub const Center: u32 = 0b0;

            /// 0b1: Down spread
            pub const Down: u32 = 0b1;
        }
    }

    /// Incrementation step
    pub mod INCSTEP {
        /// Offset (13 bits)
        pub const offset: u32 = 13;
        /// Mask (15 bits: 0x7fff << 13)
        pub const mask: u32 = 0x7fff << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// Modulation period
    pub mod MODPER {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (13 bits: 0x1fff << 0)
        pub const mask: u32 = 0x1fff << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }
}

/// PLLI2S configuration register
pub mod PLLI2SCFGR {

    /// PLLI2S division factor for I2S clocks
    pub mod PLLI2SR {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (3 bits: 0b111 << 28)
        pub const mask: u32 = 0b111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLI2S multiplication factor for VCO
    pub mod PLLI2SN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (9 bits: 0x1ff << 6)
        pub const mask: u32 = 0x1ff << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLI2S division factor for SAI1 clock
    pub mod PLLI2SQ {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (4 bits: 0b1111 << 24)
        pub const mask: u32 = 0b1111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }
}

/// RCC PLL configuration register
pub mod PLLSAICFGR {

    /// PLLSAI division factor for LCD clock
    pub mod PLLSAIR {
        /// Offset (28 bits)
        pub const offset: u32 = 28;
        /// Mask (3 bits: 0b111 << 28)
        pub const mask: u32 = 0b111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLSAI division factor for SAI1 clock
    pub mod PLLSAIQ {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (4 bits: 0b1111 << 24)
        pub const mask: u32 = 0b1111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }

    /// PLLSAI division factor for VCO
    pub mod PLLSAIN {
        /// Offset (6 bits)
        pub const offset: u32 = 6;
        /// Mask (9 bits: 0x1ff << 6)
        pub const mask: u32 = 0x1ff << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values (empty)
        pub mod RW {}
    }
}

/// RCC Dedicated Clock Configuration Register
pub mod DCKCFGR {

    /// PLLI2S division factor for SAI1 clock
    pub mod PLLI2SDIVQ {
        /// Offset (0 bits)
        pub const offset: u32 = 0;
        /// Mask (5 bits: 0b11111 << 0)
        pub const mask: u32 = 0b11111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00000: PLLI2SDIVQ = /1
            pub const Div1: u32 = 0b00000;

            /// 0b00001: PLLI2SDIVQ = /2
            pub const Div2: u32 = 0b00001;

            /// 0b00010: PLLI2SDIVQ = /3
            pub const Div3: u32 = 0b00010;

            /// 0b00011: PLLI2SDIVQ = /4
            pub const Div4: u32 = 0b00011;

            /// 0b00100: PLLI2SDIVQ = /5
            pub const Div5: u32 = 0b00100;

            /// 0b00101: PLLI2SDIVQ = /6
            pub const Div6: u32 = 0b00101;

            /// 0b00110: PLLI2SDIVQ = /7
            pub const Div7: u32 = 0b00110;

            /// 0b00111: PLLI2SDIVQ = /8
            pub const Div8: u32 = 0b00111;

            /// 0b01000: PLLI2SDIVQ = /9
            pub const Div9: u32 = 0b01000;

            /// 0b01001: PLLI2SDIVQ = /10
            pub const Div10: u32 = 0b01001;

            /// 0b01010: PLLI2SDIVQ = /11
            pub const Div11: u32 = 0b01010;

            /// 0b01011: PLLI2SDIVQ = /12
            pub const Div12: u32 = 0b01011;

            /// 0b01100: PLLI2SDIVQ = /13
            pub const Div13: u32 = 0b01100;

            /// 0b01101: PLLI2SDIVQ = /14
            pub const Div14: u32 = 0b01101;

            /// 0b01110: PLLI2SDIVQ = /15
            pub const Div15: u32 = 0b01110;

            /// 0b01111: PLLI2SDIVQ = /16
            pub const Div16: u32 = 0b01111;

            /// 0b10000: PLLI2SDIVQ = /17
            pub const Div17: u32 = 0b10000;

            /// 0b10001: PLLI2SDIVQ = /18
            pub const Div18: u32 = 0b10001;

            /// 0b10010: PLLI2SDIVQ = /19
            pub const Div19: u32 = 0b10010;

            /// 0b10011: PLLI2SDIVQ = /20
            pub const Div20: u32 = 0b10011;

            /// 0b10100: PLLI2SDIVQ = /21
            pub const Div21: u32 = 0b10100;

            /// 0b10101: PLLI2SDIVQ = /22
            pub const Div22: u32 = 0b10101;

            /// 0b10110: PLLI2SDIVQ = /23
            pub const Div23: u32 = 0b10110;

            /// 0b10111: PLLI2SDIVQ = /24
            pub const Div24: u32 = 0b10111;

            /// 0b11000: PLLI2SDIVQ = /25
            pub const Div25: u32 = 0b11000;

            /// 0b11001: PLLI2SDIVQ = /26
            pub const Div26: u32 = 0b11001;

            /// 0b11010: PLLI2SDIVQ = /27
            pub const Div27: u32 = 0b11010;

            /// 0b11011: PLLI2SDIVQ = /28
            pub const Div28: u32 = 0b11011;

            /// 0b11100: PLLI2SDIVQ = /29
            pub const Div29: u32 = 0b11100;

            /// 0b11101: PLLI2SDIVQ = /30
            pub const Div30: u32 = 0b11101;

            /// 0b11110: PLLI2SDIVQ = /31
            pub const Div31: u32 = 0b11110;

            /// 0b11111: PLLI2SDIVQ = /32
            pub const Div32: u32 = 0b11111;
        }
    }

    /// PLLSAI division factor for SAI1 clock
    pub mod PLLSAIDIVQ {
        /// Offset (8 bits)
        pub const offset: u32 = 8;
        /// Mask (5 bits: 0b11111 << 8)
        pub const mask: u32 = 0b11111 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00000: PLLSAIDIVQ = /1
            pub const Div1: u32 = 0b00000;

            /// 0b00001: PLLSAIDIVQ = /2
            pub const Div2: u32 = 0b00001;

            /// 0b00010: PLLSAIDIVQ = /3
            pub const Div3: u32 = 0b00010;

            /// 0b00011: PLLSAIDIVQ = /4
            pub const Div4: u32 = 0b00011;

            /// 0b00100: PLLSAIDIVQ = /5
            pub const Div5: u32 = 0b00100;

            /// 0b00101: PLLSAIDIVQ = /6
            pub const Div6: u32 = 0b00101;

            /// 0b00110: PLLSAIDIVQ = /7
            pub const Div7: u32 = 0b00110;

            /// 0b00111: PLLSAIDIVQ = /8
            pub const Div8: u32 = 0b00111;

            /// 0b01000: PLLSAIDIVQ = /9
            pub const Div9: u32 = 0b01000;

            /// 0b01001: PLLSAIDIVQ = /10
            pub const Div10: u32 = 0b01001;

            /// 0b01010: PLLSAIDIVQ = /11
            pub const Div11: u32 = 0b01010;

            /// 0b01011: PLLSAIDIVQ = /12
            pub const Div12: u32 = 0b01011;

            /// 0b01100: PLLSAIDIVQ = /13
            pub const Div13: u32 = 0b01100;

            /// 0b01101: PLLSAIDIVQ = /14
            pub const Div14: u32 = 0b01101;

            /// 0b01110: PLLSAIDIVQ = /15
            pub const Div15: u32 = 0b01110;

            /// 0b01111: PLLSAIDIVQ = /16
            pub const Div16: u32 = 0b01111;

            /// 0b10000: PLLSAIDIVQ = /17
            pub const Div17: u32 = 0b10000;

            /// 0b10001: PLLSAIDIVQ = /18
            pub const Div18: u32 = 0b10001;

            /// 0b10010: PLLSAIDIVQ = /19
            pub const Div19: u32 = 0b10010;

            /// 0b10011: PLLSAIDIVQ = /20
            pub const Div20: u32 = 0b10011;

            /// 0b10100: PLLSAIDIVQ = /21
            pub const Div21: u32 = 0b10100;

            /// 0b10101: PLLSAIDIVQ = /22
            pub const Div22: u32 = 0b10101;

            /// 0b10110: PLLSAIDIVQ = /23
            pub const Div23: u32 = 0b10110;

            /// 0b10111: PLLSAIDIVQ = /24
            pub const Div24: u32 = 0b10111;

            /// 0b11000: PLLSAIDIVQ = /25
            pub const Div25: u32 = 0b11000;

            /// 0b11001: PLLSAIDIVQ = /26
            pub const Div26: u32 = 0b11001;

            /// 0b11010: PLLSAIDIVQ = /27
            pub const Div27: u32 = 0b11010;

            /// 0b11011: PLLSAIDIVQ = /28
            pub const Div28: u32 = 0b11011;

            /// 0b11100: PLLSAIDIVQ = /29
            pub const Div29: u32 = 0b11100;

            /// 0b11101: PLLSAIDIVQ = /30
            pub const Div30: u32 = 0b11101;

            /// 0b11110: PLLSAIDIVQ = /31
            pub const Div31: u32 = 0b11110;

            /// 0b11111: PLLSAIDIVQ = /32
            pub const Div32: u32 = 0b11111;
        }
    }

    /// division factor for LCD_CLK
    pub mod PLLSAIDIVR {
        /// Offset (16 bits)
        pub const offset: u32 = 16;
        /// Mask (2 bits: 0b11 << 16)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: PLLSAIDIVR = /2
            pub const Div2: u32 = 0b00;

            /// 0b01: PLLSAIDIVR = /4
            pub const Div4: u32 = 0b01;

            /// 0b10: PLLSAIDIVR = /8
            pub const Div8: u32 = 0b10;

            /// 0b11: PLLSAIDIVR = /16
            pub const Div16: u32 = 0b11;
        }
    }

    /// SAI1-A clock source selection
    pub mod SAI1ASRC {
        /// Offset (20 bits)
        pub const offset: u32 = 20;
        /// Mask (2 bits: 0b11 << 20)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: SAI1-A clock frequency = f(PLLSAI_Q) / PLLSAIDIVQ
            pub const PLLSAI: u32 = 0b00;

            /// 0b01: SAI1-A clock frequency = f(PLLI2S_Q) / PLLI2SDIVQ
            pub const PLLI2S: u32 = 0b01;

            /// 0b10: SAI1-A clock frequency = Alternate function input frequency
            pub const I2S_CKIN: u32 = 0b10;
        }
    }

    /// SAI1-B clock source selection
    pub mod SAI1BSRC {
        /// Offset (22 bits)
        pub const offset: u32 = 22;
        /// Mask (2 bits: 0b11 << 22)
        pub const mask: u32 = 0b11 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b00: SAI1-B clock frequency = f(PLLSAI_Q) / PLLSAIDIVQ
            pub const PLLSAI: u32 = 0b00;

            /// 0b01: SAI1-B clock frequency = f(PLLI2S_Q) / PLLI2SDIVQ
            pub const PLLI2S: u32 = 0b01;

            /// 0b10: SAI1-B clock frequency = Alternate function input frequency
            pub const I2S_CKIN: u32 = 0b10;
        }
    }

    /// Timers clocks prescalers selection
    pub mod TIMPRE {
        /// Offset (24 bits)
        pub const offset: u32 = 24;
        /// Mask (1 bit: 1 << 24)
        pub const mask: u32 = 1 << offset;
        /// Read-only values (empty)
        pub mod R {}
        /// Write-only values (empty)
        pub mod W {}
        /// Read-write values
        pub mod RW {

            /// 0b0: If the APB prescaler is configured 1, TIMxCLK = PCLKx. Otherwise, TIMxCLK = 2xPCLKx
            pub const Mul2: u32 = 0b0;

            /// 0b1: If the APB prescaler is configured 1, 2 or 4, TIMxCLK = HCLK. Otherwise, TIMxCLK = 4xPCLKx
            pub const Mul4: u32 = 0b1;
        }
    }
}
#[repr(C)]
pub struct RegisterBlock {
    /// clock control register
    pub CR: RWRegister<u32>,

    /// PLL configuration register
    pub PLLCFGR: RWRegister<u32>,

    /// clock configuration register
    pub CFGR: RWRegister<u32>,

    /// clock interrupt register
    pub CIR: RWRegister<u32>,

    /// AHB1 peripheral reset register
    pub AHB1RSTR: RWRegister<u32>,

    /// AHB2 peripheral reset register
    pub AHB2RSTR: RWRegister<u32>,

    /// AHB3 peripheral reset register
    pub AHB3RSTR: RWRegister<u32>,

    _reserved1: [u8; 4],

    /// APB1 peripheral reset register
    pub APB1RSTR: RWRegister<u32>,

    /// APB2 peripheral reset register
    pub APB2RSTR: RWRegister<u32>,

    _reserved2: [u8; 8],

    /// AHB1 peripheral clock register
    pub AHB1ENR: RWRegister<u32>,

    /// AHB2 peripheral clock enable register
    pub AHB2ENR: RWRegister<u32>,

    /// AHB3 peripheral clock enable register
    pub AHB3ENR: RWRegister<u32>,

    _reserved3: [u8; 4],

    /// APB1 peripheral clock enable register
    pub APB1ENR: RWRegister<u32>,

    /// APB2 peripheral clock enable register
    pub APB2ENR: RWRegister<u32>,

    _reserved4: [u8; 8],

    /// AHB1 peripheral clock enable in low power mode register
    pub AHB1LPENR: RWRegister<u32>,

    /// AHB2 peripheral clock enable in low power mode register
    pub AHB2LPENR: RWRegister<u32>,

    /// AHB3 peripheral clock enable in low power mode register
    pub AHB3LPENR: RWRegister<u32>,

    _reserved5: [u8; 4],

    /// APB1 peripheral clock enable in low power mode register
    pub APB1LPENR: RWRegister<u32>,

    /// APB2 peripheral clock enabled in low power mode register
    pub APB2LPENR: RWRegister<u32>,

    _reserved6: [u8; 8],

    /// Backup domain control register
    pub BDCR: RWRegister<u32>,

    /// clock control & status register
    pub CSR: RWRegister<u32>,

    _reserved7: [u8; 8],

    /// spread spectrum clock generation register
    pub SSCGR: RWRegister<u32>,

    /// PLLI2S configuration register
    pub PLLI2SCFGR: RWRegister<u32>,

    /// RCC PLL configuration register
    pub PLLSAICFGR: RWRegister<u32>,

    /// RCC Dedicated Clock Configuration Register
    pub DCKCFGR: RWRegister<u32>,
}
pub struct ResetValues {
    pub CR: u32,
    pub PLLCFGR: u32,
    pub CFGR: u32,
    pub CIR: u32,
    pub AHB1RSTR: u32,
    pub AHB2RSTR: u32,
    pub AHB3RSTR: u32,
    pub APB1RSTR: u32,
    pub APB2RSTR: u32,
    pub AHB1ENR: u32,
    pub AHB2ENR: u32,
    pub AHB3ENR: u32,
    pub APB1ENR: u32,
    pub APB2ENR: u32,
    pub AHB1LPENR: u32,
    pub AHB2LPENR: u32,
    pub AHB3LPENR: u32,
    pub APB1LPENR: u32,
    pub APB2LPENR: u32,
    pub BDCR: u32,
    pub CSR: u32,
    pub SSCGR: u32,
    pub PLLI2SCFGR: u32,
    pub PLLSAICFGR: u32,
    pub DCKCFGR: u32,
}
#[cfg(not(feature = "nosync"))]
pub struct Instance {
    pub(crate) addr: u32,
    pub(crate) _marker: PhantomData<*const RegisterBlock>,
}
#[cfg(not(feature = "nosync"))]
impl ::core::ops::Deref for Instance {
    type Target = RegisterBlock;
    #[inline(always)]
    fn deref(&self) -> &RegisterBlock {
        unsafe { &*(self.addr as *const _) }
    }
}
#[cfg(feature = "rtic")]
unsafe impl Send for Instance {}

/// Access functions for the RCC peripheral instance
pub mod RCC {
    use super::ResetValues;

    #[cfg(not(feature = "nosync"))]
    use super::Instance;

    #[cfg(not(feature = "nosync"))]
    const INSTANCE: Instance = Instance {
        addr: 0x40023800,
        _marker: ::core::marker::PhantomData,
    };

    /// Reset values for each field in RCC
    pub const reset: ResetValues = ResetValues {
        CR: 0x00000083,
        PLLCFGR: 0x24003010,
        CFGR: 0x00000000,
        CIR: 0x00000000,
        AHB1RSTR: 0x00000000,
        AHB2RSTR: 0x00000000,
        AHB3RSTR: 0x00000000,
        APB1RSTR: 0x00000000,
        APB2RSTR: 0x00000000,
        AHB1ENR: 0x00100000,
        AHB2ENR: 0x00000000,
        AHB3ENR: 0x00000000,
        APB1ENR: 0x00000000,
        APB2ENR: 0x00000000,
        AHB1LPENR: 0x7E6791FF,
        AHB2LPENR: 0x000000F1,
        AHB3LPENR: 0x00000001,
        APB1LPENR: 0x36FEC9FF,
        APB2LPENR: 0x00075F33,
        BDCR: 0x00000000,
        CSR: 0x0E000000,
        SSCGR: 0x00000000,
        PLLI2SCFGR: 0x20003000,
        PLLSAICFGR: 0x24003000,
        DCKCFGR: 0x00000000,
    };

    #[cfg(not(feature = "nosync"))]
    #[allow(renamed_and_removed_lints)]
    #[allow(private_no_mangle_statics)]
    #[no_mangle]
    static mut RCC_TAKEN: bool = false;

    /// Safe access to RCC
    ///
    /// This function returns `Some(Instance)` if this instance is not
    /// currently taken, and `None` if it is. This ensures that if you
    /// do get `Some(Instance)`, you are ensured unique access to
    /// the peripheral and there cannot be data races (unless other
    /// code uses `unsafe`, of course). You can then pass the
    /// `Instance` around to other functions as required. When you're
    /// done with it, you can call `release(instance)` to return it.
    ///
    /// `Instance` itself dereferences to a `RegisterBlock`, which
    /// provides access to the peripheral's registers.
    #[cfg(not(feature = "nosync"))]
    #[inline]
    pub fn take() -> Option<Instance> {
        external_cortex_m::interrupt::free(|_| unsafe {
            if RCC_TAKEN {
                None
            } else {
                RCC_TAKEN = true;
                Some(INSTANCE)
            }
        })
    }

    /// Release exclusive access to RCC
    ///
    /// This function allows you to return an `Instance` so that it
    /// is available to `take()` again. This function will panic if
    /// you return a different `Instance` or if this instance is not
    /// already taken.
    #[cfg(not(feature = "nosync"))]
    #[inline]
    pub fn release(inst: Instance) {
        external_cortex_m::interrupt::free(|_| unsafe {
            if RCC_TAKEN && inst.addr == INSTANCE.addr {
                RCC_TAKEN = false;
            } else {
                panic!("Released a peripheral which was not taken");
            }
        });
    }

    /// Unsafely steal RCC
    ///
    /// This function is similar to take() but forcibly takes the
    /// Instance, marking it as taken irregardless of its previous
    /// state.
    #[cfg(not(feature = "nosync"))]
    #[inline]
    pub unsafe fn steal() -> Instance {
        RCC_TAKEN = true;
        INSTANCE
    }
}

/// Raw pointer to RCC
///
/// Dereferencing this is unsafe because you are not ensured unique
/// access to the peripheral, so you may encounter data races with
/// other users of this peripheral. It is up to you to ensure you
/// will not cause data races.
///
/// This constant is provided for ease of use in unsafe code: you can
/// simply call for example `write_reg!(gpio, GPIOA, ODR, 1);`.
pub const RCC: *const RegisterBlock = 0x40023800 as *const _;
