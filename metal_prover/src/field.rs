use field::{Mersenne31Complex, Mersenne31Field, Mersenne31Quartic};

pub type BaseField = Mersenne31Field;
pub type Ext2Field = Mersenne31Complex;
pub type Ext4Field = Mersenne31Quartic;

/// Repr(C) base field type matching the Metal shader's BF layout.
/// A single u32 limb holding a value in [0, 2^31 - 1].
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalBaseField {
    pub limb: u32,
}

/// Repr(C) quadratic extension field (Mersenne31Complex) matching the Metal shader layout.
/// Two base-field coefficients, 8-byte aligned.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalExt2Field {
    pub coefficients: [MetalBaseField; 2],
}

/// Repr(C) quartic extension field matching the Metal shader layout.
/// Two Ext2 coefficients, 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalExt4Field {
    pub coefficients: [MetalExt2Field; 2],
}

impl From<Mersenne31Field> for MetalBaseField {
    fn from(f: Mersenne31Field) -> Self {
        Self { limb: f.0 }
    }
}

impl From<MetalBaseField> for Mersenne31Field {
    fn from(f: MetalBaseField) -> Self {
        Self(f.limb)
    }
}

impl From<Mersenne31Complex> for MetalExt2Field {
    fn from(f: Mersenne31Complex) -> Self {
        Self {
            coefficients: [
                MetalBaseField { limb: f.c0.0 },
                MetalBaseField { limb: f.c1.0 },
            ],
        }
    }
}

impl From<MetalExt2Field> for Mersenne31Complex {
    fn from(f: MetalExt2Field) -> Self {
        Self {
            c0: Mersenne31Field(f.coefficients[0].limb),
            c1: Mersenne31Field(f.coefficients[1].limb),
        }
    }
}
