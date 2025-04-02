// Repository: Infineon/traveo-t2g-pal
// File: pal_crates/cyt4en_a/src/m0/cpuss/cm7_0_status.rs

#[doc = "Register `CM7_0_STATUS` reader"]
pub struct R(crate::R<CM7_0_STATUS_SPEC>);
impl core::ops::Deref for R {
    type Target = crate::R<CM7_0_STATUS_SPEC>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<crate::R<CM7_0_STATUS_SPEC>> for R {
    #[inline(always)]
    fn from(reader: crate::R<CM7_0_STATUS_SPEC>) -> Self {
        R(reader)
    }
}
#[doc = "Field `SLEEPING` reader - Specifies if the CPU is in Active, Sleep or DeepSleep power mode: - Active power mode: SLEEPING is '0'. - Sleep power mode: SLEEPING is '1' and SLEEPDEEP is '0'. - DeepSleep power mode: SLEEPING is '1' and SLEEPDEEP is '1'."]
pub type SLEEPING_R = crate::BitReader<bool>;
#[doc = "Field `SLEEPDEEP` reader - Specifies if the CPU is in Sleep or DeepSleep power mode. See SLEEPING field."]
pub type SLEEPDEEP_R = crate::BitReader<bool>;
#[doc = "Field `PWR_DONE` reader - After a PWR_MODE change this flag indicates if the new power mode has taken effect or not. Note: this flag can also change as a result of a change in debug power up req"]
pub type PWR_DONE_R = crate::BitReader<bool>;
#[doc = "Field `TCMC_CM7_1_MS` reader - Outstanding transactions from CM7 1. '0': There are no outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from CM7 1. '1': There are outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from CM7 1. Do not switch the CM7 power off."]
pub type TCMC_CM7_1_MS_R = crate::BitReader<bool>;
#[doc = "Field `TCMC_EXT_MS_2_TO_0` reader - Outstanding transactions from external AXI masters 2 to 0. '0': There are no outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI masters 2 to 0. '1': There are outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI masters 2 to 0. Do not switch the CM7 power off."]
pub type TCMC_EXT_MS_2_TO_0_R = crate::BitReader<bool>;
#[doc = "Field `TCMC_EXT_MS_3` reader - Outstanding transactions from external AXI master 3. '0': There are no outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI master 3. '1': There are outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI master 3. Do not switch the CM7 power off."]
pub type TCMC_EXT_MS_3_R = crate::BitReader<bool>;
#[doc = "Field `TCMC_AHB_MS` reader - Ongoing AHB transaction. '0': There is no ongoing transaction to the CM7 I/D-TCM slave port (AHBS) from an AHB master. '1': There is an ongoing transaction to the CM7 I/D-TCM slave port (AHBS) from an AHB master. Do not switch the CM7 power off."]
pub type TCMC_AHB_MS_R = crate::BitReader<bool>;
impl R {
    #[doc = "Bit 0 - Specifies if the CPU is in Active, Sleep or DeepSleep power mode: - Active power mode: SLEEPING is '0'. - Sleep power mode: SLEEPING is '1' and SLEEPDEEP is '0'. - DeepSleep power mode: SLEEPING is '1' and SLEEPDEEP is '1'."]
    #[inline(always)]
    pub fn sleeping(&self) -> SLEEPING_R {
        SLEEPING_R::new((self.bits & 1) != 0)
    }
    #[doc = "Bit 1 - Specifies if the CPU is in Sleep or DeepSleep power mode. See SLEEPING field."]
    #[inline(always)]
    pub fn sleepdeep(&self) -> SLEEPDEEP_R {
        SLEEPDEEP_R::new(((self.bits >> 1) & 1) != 0)
    }
    #[doc = "Bit 4 - After a PWR_MODE change this flag indicates if the new power mode has taken effect or not. Note: this flag can also change as a result of a change in debug power up req"]
    #[inline(always)]
    pub fn pwr_done(&self) -> PWR_DONE_R {
        PWR_DONE_R::new(((self.bits >> 4) & 1) != 0)
    }
    #[doc = "Bit 9 - Outstanding transactions from CM7 1. '0': There are no outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from CM7 1. '1': There are outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from CM7 1. Do not switch the CM7 power off."]
    #[inline(always)]
    pub fn tcmc_cm7_1_ms(&self) -> TCMC_CM7_1_MS_R {
        TCMC_CM7_1_MS_R::new(((self.bits >> 9) & 1) != 0)
    }
    #[doc = "Bit 10 - Outstanding transactions from external AXI masters 2 to 0. '0': There are no outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI masters 2 to 0. '1': There are outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI masters 2 to 0. Do not switch the CM7 power off."]
    #[inline(always)]
    pub fn tcmc_ext_ms_2_to_0(&self) -> TCMC_EXT_MS_2_TO_0_R {
        TCMC_EXT_MS_2_TO_0_R::new(((self.bits >> 10) & 1) != 0)
    }
    #[doc = "Bit 11 - Outstanding transactions from external AXI master 3. '0': There are no outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI master 3. '1': There are outstanding transactions to the CM7 I/D-TCM slave port (AHBS) from external AXI master 3. Do not switch the CM7 power off."]
    #[inline(always)]
    pub fn tcmc_ext_ms_3(&self) -> TCMC_EXT_MS_3_R {
        TCMC_EXT_MS_3_R::new(((self.bits >> 11) & 1) != 0)
    }
    #[doc = "Bit 12 - Ongoing AHB transaction. '0': There is no ongoing transaction to the CM7 I/D-TCM slave port (AHBS) from an AHB master. '1': There is an ongoing transaction to the CM7 I/D-TCM slave port (AHBS) from an AHB master. Do not switch the CM7 power off."]
    #[inline(always)]
    pub fn tcmc_ahb_ms(&self) -> TCMC_AHB_MS_R {
        TCMC_AHB_MS_R::new(((self.bits >> 12) & 1) != 0)
    }
}
#[doc = "CM7 0 status\n\nThis register you can [`read`](crate::generic::Reg::read). See [API](https://docs.rs/svd2rust/#read--modify--write-api).\n\nFor information about available fields see [cm7_0_status](index.html) module"]
pub struct CM7_0_STATUS_SPEC;
impl crate::RegisterSpec for CM7_0_STATUS_SPEC {
    type Ux = u32;
}
#[doc = "`read()` method returns [cm7_0_status::R](R) reader structure"]
impl crate::Readable for CM7_0_STATUS_SPEC {
    type Reader = R;
}
#[doc = "`reset()` method sets CM7_0_STATUS to value 0x13"]
impl crate::Resettable for CM7_0_STATUS_SPEC {
    const RESET_VALUE: Self::Ux = 0x13;
}
