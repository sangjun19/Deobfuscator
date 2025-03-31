// Repository: LennyPhoenix/stm32l476
// File: src/tsc.rs

#[repr(C)]
#[doc = "Register block"]
pub struct RegisterBlock {
    cr: Cr,
    ier: Ier,
    icr: Icr,
    isr: Isr,
    iohcr: Iohcr,
    _reserved5: [u8; 0x04],
    ioascr: Ioascr,
    _reserved6: [u8; 0x04],
    ioscr: Ioscr,
    _reserved7: [u8; 0x04],
    ioccr: Ioccr,
    _reserved8: [u8; 0x04],
    iogcsr: Iogcsr,
    iog1cr: Iog1cr,
    iog2cr: Iog2cr,
    iog3cr: Iog3cr,
    iog4cr: Iog4cr,
    iog5cr: Iog5cr,
    iog6cr: Iog6cr,
    iog7cr: Iog7cr,
    iog8cr: Iog8cr,
}
impl RegisterBlock {
    #[doc = "0x00 - control register"]
    #[inline(always)]
    pub const fn cr(&self) -> &Cr {
        &self.cr
    }
    #[doc = "0x04 - interrupt enable register"]
    #[inline(always)]
    pub const fn ier(&self) -> &Ier {
        &self.ier
    }
    #[doc = "0x08 - interrupt clear register"]
    #[inline(always)]
    pub const fn icr(&self) -> &Icr {
        &self.icr
    }
    #[doc = "0x0c - interrupt status register"]
    #[inline(always)]
    pub const fn isr(&self) -> &Isr {
        &self.isr
    }
    #[doc = "0x10 - I/O hysteresis control register"]
    #[inline(always)]
    pub const fn iohcr(&self) -> &Iohcr {
        &self.iohcr
    }
    #[doc = "0x18 - I/O analog switch control register"]
    #[inline(always)]
    pub const fn ioascr(&self) -> &Ioascr {
        &self.ioascr
    }
    #[doc = "0x20 - I/O sampling control register"]
    #[inline(always)]
    pub const fn ioscr(&self) -> &Ioscr {
        &self.ioscr
    }
    #[doc = "0x28 - I/O channel control register"]
    #[inline(always)]
    pub const fn ioccr(&self) -> &Ioccr {
        &self.ioccr
    }
    #[doc = "0x30 - I/O group control status register"]
    #[inline(always)]
    pub const fn iogcsr(&self) -> &Iogcsr {
        &self.iogcsr
    }
    #[doc = "0x34 - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog1cr(&self) -> &Iog1cr {
        &self.iog1cr
    }
    #[doc = "0x38 - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog2cr(&self) -> &Iog2cr {
        &self.iog2cr
    }
    #[doc = "0x3c - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog3cr(&self) -> &Iog3cr {
        &self.iog3cr
    }
    #[doc = "0x40 - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog4cr(&self) -> &Iog4cr {
        &self.iog4cr
    }
    #[doc = "0x44 - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog5cr(&self) -> &Iog5cr {
        &self.iog5cr
    }
    #[doc = "0x48 - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog6cr(&self) -> &Iog6cr {
        &self.iog6cr
    }
    #[doc = "0x4c - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog7cr(&self) -> &Iog7cr {
        &self.iog7cr
    }
    #[doc = "0x50 - I/O group x counter register"]
    #[inline(always)]
    pub const fn iog8cr(&self) -> &Iog8cr {
        &self.iog8cr
    }
}
#[doc = "CR (rw) register accessor: control register\n\nYou can [`read`](crate::Reg::read) this register and get [`cr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`cr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@cr`] module"]
#[doc(alias = "CR")]
pub type Cr = crate::Reg<cr::CrSpec>;
#[doc = "control register"]
pub mod cr;
#[doc = "IER (rw) register accessor: interrupt enable register\n\nYou can [`read`](crate::Reg::read) this register and get [`ier::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`ier::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@ier`] module"]
#[doc(alias = "IER")]
pub type Ier = crate::Reg<ier::IerSpec>;
#[doc = "interrupt enable register"]
pub mod ier;
#[doc = "ICR (rw) register accessor: interrupt clear register\n\nYou can [`read`](crate::Reg::read) this register and get [`icr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`icr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@icr`] module"]
#[doc(alias = "ICR")]
pub type Icr = crate::Reg<icr::IcrSpec>;
#[doc = "interrupt clear register"]
pub mod icr;
#[doc = "ISR (rw) register accessor: interrupt status register\n\nYou can [`read`](crate::Reg::read) this register and get [`isr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`isr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@isr`] module"]
#[doc(alias = "ISR")]
pub type Isr = crate::Reg<isr::IsrSpec>;
#[doc = "interrupt status register"]
pub mod isr;
#[doc = "IOHCR (rw) register accessor: I/O hysteresis control register\n\nYou can [`read`](crate::Reg::read) this register and get [`iohcr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`iohcr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iohcr`] module"]
#[doc(alias = "IOHCR")]
pub type Iohcr = crate::Reg<iohcr::IohcrSpec>;
#[doc = "I/O hysteresis control register"]
pub mod iohcr;
#[doc = "IOASCR (rw) register accessor: I/O analog switch control register\n\nYou can [`read`](crate::Reg::read) this register and get [`ioascr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`ioascr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@ioascr`] module"]
#[doc(alias = "IOASCR")]
pub type Ioascr = crate::Reg<ioascr::IoascrSpec>;
#[doc = "I/O analog switch control register"]
pub mod ioascr;
#[doc = "IOSCR (rw) register accessor: I/O sampling control register\n\nYou can [`read`](crate::Reg::read) this register and get [`ioscr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`ioscr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@ioscr`] module"]
#[doc(alias = "IOSCR")]
pub type Ioscr = crate::Reg<ioscr::IoscrSpec>;
#[doc = "I/O sampling control register"]
pub mod ioscr;
#[doc = "IOCCR (rw) register accessor: I/O channel control register\n\nYou can [`read`](crate::Reg::read) this register and get [`ioccr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`ioccr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@ioccr`] module"]
#[doc(alias = "IOCCR")]
pub type Ioccr = crate::Reg<ioccr::IoccrSpec>;
#[doc = "I/O channel control register"]
pub mod ioccr;
#[doc = "IOGCSR (rw) register accessor: I/O group control status register\n\nYou can [`read`](crate::Reg::read) this register and get [`iogcsr::R`]. You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`iogcsr::W`]. You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iogcsr`] module"]
#[doc(alias = "IOGCSR")]
pub type Iogcsr = crate::Reg<iogcsr::IogcsrSpec>;
#[doc = "I/O group control status register"]
pub mod iogcsr;
#[doc = "IOG1CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog1cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog1cr`] module"]
#[doc(alias = "IOG1CR")]
pub type Iog1cr = crate::Reg<iog1cr::Iog1crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog1cr;
#[doc = "IOG2CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog2cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog2cr`] module"]
#[doc(alias = "IOG2CR")]
pub type Iog2cr = crate::Reg<iog2cr::Iog2crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog2cr;
#[doc = "IOG3CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog3cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog3cr`] module"]
#[doc(alias = "IOG3CR")]
pub type Iog3cr = crate::Reg<iog3cr::Iog3crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog3cr;
#[doc = "IOG4CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog4cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog4cr`] module"]
#[doc(alias = "IOG4CR")]
pub type Iog4cr = crate::Reg<iog4cr::Iog4crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog4cr;
#[doc = "IOG5CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog5cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog5cr`] module"]
#[doc(alias = "IOG5CR")]
pub type Iog5cr = crate::Reg<iog5cr::Iog5crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog5cr;
#[doc = "IOG6CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog6cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog6cr`] module"]
#[doc(alias = "IOG6CR")]
pub type Iog6cr = crate::Reg<iog6cr::Iog6crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog6cr;
#[doc = "IOG7CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog7cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog7cr`] module"]
#[doc(alias = "IOG7CR")]
pub type Iog7cr = crate::Reg<iog7cr::Iog7crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog7cr;
#[doc = "IOG8CR (r) register accessor: I/O group x counter register\n\nYou can [`read`](crate::Reg::read) this register and get [`iog8cr::R`]. See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [`mod@iog8cr`] module"]
#[doc(alias = "IOG8CR")]
pub type Iog8cr = crate::Reg<iog8cr::Iog8crSpec>;
#[doc = "I/O group x counter register"]
pub mod iog8cr;
