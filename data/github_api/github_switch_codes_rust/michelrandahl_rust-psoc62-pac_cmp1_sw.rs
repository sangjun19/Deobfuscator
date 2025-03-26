// Repository: michelrandahl/rust-psoc62-pac
// File: src/lpcomp/cmp1_sw.rs

#[doc = "Register `CMP1_SW` reader"]
pub type R = crate::R<CMP1_SW_SPEC>;
#[doc = "Register `CMP1_SW` writer"]
pub type W = crate::W<CMP1_SW_SPEC>;
#[doc = "Field `CMP1_IP1` reader - Comparator 1 positive terminal isolation switch to GPIO"]
pub type CMP1_IP1_R = crate::BitReader;
#[doc = "Field `CMP1_IP1` writer - Comparator 1 positive terminal isolation switch to GPIO"]
pub type CMP1_IP1_W<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `CMP1_AP1` reader - Comparator 1 positive terminal switch to amuxbusA"]
pub type CMP1_AP1_R = crate::BitReader;
#[doc = "Field `CMP1_AP1` writer - Comparator 1 positive terminal switch to amuxbusA"]
pub type CMP1_AP1_W<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `CMP1_BP1` reader - Comparator 1 positive terminal switch to amuxbusB"]
pub type CMP1_BP1_R = crate::BitReader;
#[doc = "Field `CMP1_BP1` writer - Comparator 1 positive terminal switch to amuxbusB"]
pub type CMP1_BP1_W<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `CMP1_IN1` reader - Comparator 1 negative terminal isolation switch to GPIO"]
pub type CMP1_IN1_R = crate::BitReader;
#[doc = "Field `CMP1_IN1` writer - Comparator 1 negative terminal isolation switch to GPIO"]
pub type CMP1_IN1_W<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `CMP1_AN1` reader - Comparator 1 negative terminal switch to amuxbusA"]
pub type CMP1_AN1_R = crate::BitReader;
#[doc = "Field `CMP1_AN1` writer - Comparator 1 negative terminal switch to amuxbusA"]
pub type CMP1_AN1_W<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `CMP1_BN1` reader - Comparator 1 negative terminal switch to amuxbusB"]
pub type CMP1_BN1_R = crate::BitReader;
#[doc = "Field `CMP1_BN1` writer - Comparator 1 negative terminal switch to amuxbusB"]
pub type CMP1_BN1_W<'a, REG> = crate::BitWriter<'a, REG>;
#[doc = "Field `CMP1_VN1` reader - Comparator 1 negative terminal switch to local Vref (LPREF_EN must be set)"]
pub type CMP1_VN1_R = crate::BitReader;
#[doc = "Field `CMP1_VN1` writer - Comparator 1 negative terminal switch to local Vref (LPREF_EN must be set)"]
pub type CMP1_VN1_W<'a, REG> = crate::BitWriter<'a, REG>;
impl R {
    #[doc = "Bit 0 - Comparator 1 positive terminal isolation switch to GPIO"]
    #[inline(always)]
    pub fn cmp1_ip1(&self) -> CMP1_IP1_R {
        CMP1_IP1_R::new((self.bits & 1) != 0)
    }
    #[doc = "Bit 1 - Comparator 1 positive terminal switch to amuxbusA"]
    #[inline(always)]
    pub fn cmp1_ap1(&self) -> CMP1_AP1_R {
        CMP1_AP1_R::new(((self.bits >> 1) & 1) != 0)
    }
    #[doc = "Bit 2 - Comparator 1 positive terminal switch to amuxbusB"]
    #[inline(always)]
    pub fn cmp1_bp1(&self) -> CMP1_BP1_R {
        CMP1_BP1_R::new(((self.bits >> 2) & 1) != 0)
    }
    #[doc = "Bit 4 - Comparator 1 negative terminal isolation switch to GPIO"]
    #[inline(always)]
    pub fn cmp1_in1(&self) -> CMP1_IN1_R {
        CMP1_IN1_R::new(((self.bits >> 4) & 1) != 0)
    }
    #[doc = "Bit 5 - Comparator 1 negative terminal switch to amuxbusA"]
    #[inline(always)]
    pub fn cmp1_an1(&self) -> CMP1_AN1_R {
        CMP1_AN1_R::new(((self.bits >> 5) & 1) != 0)
    }
    #[doc = "Bit 6 - Comparator 1 negative terminal switch to amuxbusB"]
    #[inline(always)]
    pub fn cmp1_bn1(&self) -> CMP1_BN1_R {
        CMP1_BN1_R::new(((self.bits >> 6) & 1) != 0)
    }
    #[doc = "Bit 7 - Comparator 1 negative terminal switch to local Vref (LPREF_EN must be set)"]
    #[inline(always)]
    pub fn cmp1_vn1(&self) -> CMP1_VN1_R {
        CMP1_VN1_R::new(((self.bits >> 7) & 1) != 0)
    }
}
impl W {
    #[doc = "Bit 0 - Comparator 1 positive terminal isolation switch to GPIO"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_ip1(&mut self) -> CMP1_IP1_W<CMP1_SW_SPEC> {
        CMP1_IP1_W::new(self, 0)
    }
    #[doc = "Bit 1 - Comparator 1 positive terminal switch to amuxbusA"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_ap1(&mut self) -> CMP1_AP1_W<CMP1_SW_SPEC> {
        CMP1_AP1_W::new(self, 1)
    }
    #[doc = "Bit 2 - Comparator 1 positive terminal switch to amuxbusB"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_bp1(&mut self) -> CMP1_BP1_W<CMP1_SW_SPEC> {
        CMP1_BP1_W::new(self, 2)
    }
    #[doc = "Bit 4 - Comparator 1 negative terminal isolation switch to GPIO"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_in1(&mut self) -> CMP1_IN1_W<CMP1_SW_SPEC> {
        CMP1_IN1_W::new(self, 4)
    }
    #[doc = "Bit 5 - Comparator 1 negative terminal switch to amuxbusA"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_an1(&mut self) -> CMP1_AN1_W<CMP1_SW_SPEC> {
        CMP1_AN1_W::new(self, 5)
    }
    #[doc = "Bit 6 - Comparator 1 negative terminal switch to amuxbusB"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_bn1(&mut self) -> CMP1_BN1_W<CMP1_SW_SPEC> {
        CMP1_BN1_W::new(self, 6)
    }
    #[doc = "Bit 7 - Comparator 1 negative terminal switch to local Vref (LPREF_EN must be set)"]
    #[inline(always)]
    #[must_use]
    pub fn cmp1_vn1(&mut self) -> CMP1_VN1_W<CMP1_SW_SPEC> {
        CMP1_VN1_W::new(self, 7)
    }
    #[doc = r" Writes raw bits to the register."]
    #[doc = r""]
    #[doc = r" # Safety"]
    #[doc = r""]
    #[doc = r" Passing incorrect value can cause undefined behaviour. See reference manual"]
    #[inline(always)]
    pub unsafe fn bits(&mut self, bits: u32) -> &mut Self {
        self.bits = bits;
        self
    }
}
#[doc = "Comparator 1 switch control\n\nYou can [`read`](crate::generic::Reg::read) this register and get [`cmp1_sw::R`](R).  You can [`reset`](crate::generic::Reg::reset), [`write`](crate::generic::Reg::write), [`write_with_zero`](crate::generic::Reg::write_with_zero) this register using [`cmp1_sw::W`](W). You can also [`modify`](crate::generic::Reg::modify) this register. See [API](https://docs.rs/svd2rust/#read--modify--write-api)."]
pub struct CMP1_SW_SPEC;
impl crate::RegisterSpec for CMP1_SW_SPEC {
    type Ux = u32;
}
#[doc = "`read()` method returns [`cmp1_sw::R`](R) reader structure"]
impl crate::Readable for CMP1_SW_SPEC {}
#[doc = "`write(|w| ..)` method takes [`cmp1_sw::W`](W) writer structure"]
impl crate::Writable for CMP1_SW_SPEC {
    const ZERO_TO_MODIFY_FIELDS_BITMAP: Self::Ux = 0;
    const ONE_TO_MODIFY_FIELDS_BITMAP: Self::Ux = 0;
}
#[doc = "`reset()` method sets CMP1_SW to value 0"]
impl crate::Resettable for CMP1_SW_SPEC {
    const RESET_VALUE: Self::Ux = 0;
}
