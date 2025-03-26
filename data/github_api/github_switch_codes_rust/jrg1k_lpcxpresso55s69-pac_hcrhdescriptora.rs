// Repository: jrg1k/lpcxpresso55s69-pac
// File: src/usbfsh/hcrhdescriptora.rs

#[doc = "Register `HCRHDESCRIPTORA` reader"]
pub type R = crate::R<HcrhdescriptoraSpec>;
#[doc = "Register `HCRHDESCRIPTORA` writer"]
pub type W = crate::W<HcrhdescriptoraSpec>;
#[doc = "Field `NDP` reader - NumberDownstreamPorts These bits specify the number of downstream ports supported by the root hub."]
pub type NdpR = crate::FieldReader;
#[doc = "Field `NDP` writer - NumberDownstreamPorts These bits specify the number of downstream ports supported by the root hub."]
pub type NdpW<'a, REG> = crate::FieldWriter<'a, REG, 8>;
#[doc = "Field `PSM` reader - PowerSwitchingMode This bit is used to specify how the power switching of the root hub ports is controlled."]
pub type PsmR = crate::BitReader;
#[doc = "Field `PSM` writer - PowerSwitchingMode This bit is used to specify how the power switching of the root hub ports is controlled."]
pub type PsmW<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `NPS` reader - NoPowerSwitching These bits are used to specify whether power switching is supported or port are always powered."]
pub type NpsR = crate::BitReader;
#[doc = "Field `NPS` writer - NoPowerSwitching These bits are used to specify whether power switching is supported or port are always powered."]
pub type NpsW<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `DT` reader - DeviceType This bit specifies that the root hub is not a compound device."]
pub type DtR = crate::BitReader;
#[doc = "Field `DT` writer - DeviceType This bit specifies that the root hub is not a compound device."]
pub type DtW<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `OCPM` reader - OverCurrentProtectionMode This bit describes how the overcurrent status for the root hub ports are reported."]
pub type OcpmR = crate::BitReader;
#[doc = "Field `OCPM` writer - OverCurrentProtectionMode This bit describes how the overcurrent status for the root hub ports are reported."]
pub type OcpmW<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `NOCP` reader - NoOverCurrentProtection This bit describes how the overcurrent status for the root hub ports are reported."]
pub type NocpR = crate::BitReader;
#[doc = "Field `NOCP` writer - NoOverCurrentProtection This bit describes how the overcurrent status for the root hub ports are reported."]
pub type NocpW<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `POTPGT` reader - PowerOnToPowerGoodTime This byte specifies the duration the HCD has to wait before accessing a powered-on port of the root hub."]
pub type PotpgtR = crate::FieldReader;
#[doc = "Field `POTPGT` writer - PowerOnToPowerGoodTime This byte specifies the duration the HCD has to wait before accessing a powered-on port of the root hub."]
pub type PotpgtW<'a, REG> = crate::FieldWriter<'a, REG, 8>;
impl R {
    #[doc = "Bits 0:7 - NumberDownstreamPorts These bits specify the number of downstream ports supported by the root hub."]
    #[inline(always)]
    pub fn ndp(&self) -> NdpR {
        NdpR::new((self.bits & 0xff) as u8)
    }
    #[doc = "Bit 8 - PowerSwitchingMode This bit is used to specify how the power switching of the root hub ports is controlled."]
    #[inline(always)]
    pub fn psm(&self) -> PsmR {
        PsmR::new(((self.bits >> 8) & 1) != 0)
    }
    #[doc = "Bit 9 - NoPowerSwitching These bits are used to specify whether power switching is supported or port are always powered."]
    #[inline(always)]
    pub fn nps(&self) -> NpsR {
        NpsR::new(((self.bits >> 9) & 1) != 0)
    }
    #[doc = "Bit 10 - DeviceType This bit specifies that the root hub is not a compound device."]
    #[inline(always)]
    pub fn dt(&self) -> DtR {
        DtR::new(((self.bits >> 10) & 1) != 0)
    }
    #[doc = "Bit 11 - OverCurrentProtectionMode This bit describes how the overcurrent status for the root hub ports are reported."]
    #[inline(always)]
    pub fn ocpm(&self) -> OcpmR {
        OcpmR::new(((self.bits >> 11) & 1) != 0)
    }
    #[doc = "Bit 12 - NoOverCurrentProtection This bit describes how the overcurrent status for the root hub ports are reported."]
    #[inline(always)]
    pub fn nocp(&self) -> NocpR {
        NocpR::new(((self.bits >> 12) & 1) != 0)
    }
    #[doc = "Bits 24:31 - PowerOnToPowerGoodTime This byte specifies the duration the HCD has to wait before accessing a powered-on port of the root hub."]
    #[inline(always)]
    pub fn potpgt(&self) -> PotpgtR {
        PotpgtR::new(((self.bits >> 24) & 0xff) as u8)
    }
}
impl W {
    #[doc = "Bits 0:7 - NumberDownstreamPorts These bits specify the number of downstream ports supported by the root hub."]
    #[inline(always)]
    pub fn ndp(&mut self) -> NdpW<HcrhdescriptoraSpec> {
        NdpW::new(self, 0)
    }
    #[doc = "Bit 8 - PowerSwitchingMode This bit is used to specify how the power switching of the root hub ports is controlled."]
    #[inline(always)]
    pub fn psm(&mut self) -> PsmW<HcrhdescriptoraSpec> {
        PsmW::new(self, 8)
    }
    #[doc = "Bit 9 - NoPowerSwitching These bits are used to specify whether power switching is supported or port are always powered."]
    #[inline(always)]
    pub fn nps(&mut self) -> NpsW<HcrhdescriptoraSpec> {
        NpsW::new(self, 9)
    }
    #[doc = "Bit 10 - DeviceType This bit specifies that the root hub is not a compound device."]
    #[inline(always)]
    pub fn dt(&mut self) -> DtW<HcrhdescriptoraSpec> {
        DtW::new(self, 10)
    }
    #[doc = "Bit 11 - OverCurrentProtectionMode This bit describes how the overcurrent status for the root hub ports are reported."]
    #[inline(always)]
    pub fn ocpm(&mut self) -> OcpmW<HcrhdescriptoraSpec> {
        OcpmW::new(self, 11)
    }
    #[doc = "Bit 12 - NoOverCurrentProtection This bit describes how the overcurrent status for the root hub ports are reported."]
    #[inline(always)]
    pub fn nocp(&mut self) -> NocpW<HcrhdescriptoraSpec> {
        NocpW::new(self, 12)
    }
    #[doc = "Bits 24:31 - PowerOnToPowerGoodTime This byte specifies the duration the HCD has to wait before accessing a powered-on port of the root hub."]
    #[inline(always)]
    pub fn potpgt(&mut self) -> PotpgtW<HcrhdescriptoraSpec> {
        PotpgtW::new(self, 24)
    }
}
#[doc = "First of the two registers which describes the characteristics of the root hub\n\nYou can [`read`](crate::Reg::read) this register and get [`hcrhdescriptora::R`](R). You can [`reset`](crate::Reg::reset), [`write`](crate::Reg::write), [`write_with_zero`](crate::Reg::write_with_zero) this register using [`hcrhdescriptora::W`](W). You can also [`modify`](crate::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api)."]
pub struct HcrhdescriptoraSpec;
impl crate::RegisterSpec for HcrhdescriptoraSpec {
    type Ux = u32;
}
#[doc = "`read()` method returns [`hcrhdescriptora::R`](R) reader structure"]
impl crate::Readable for HcrhdescriptoraSpec {}
#[doc = "`write(|w| ..)` method takes [`hcrhdescriptora::W`](W) writer structure"]
impl crate::Writable for HcrhdescriptoraSpec {
    type Safety = crate::Unsafe;
    const ZERO_TO_MODIFY_FIELDS_BITMAP: u32 = 0;
    const ONE_TO_MODIFY_FIELDS_BITMAP: u32 = 0;
}
#[doc = "`reset()` method sets HCRHDESCRIPTORA to value 0xff00_0902"]
impl crate::Resettable for HcrhdescriptoraSpec {
    const RESET_VALUE: u32 = 0xff00_0902;
}
