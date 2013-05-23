package opt

import (
	. "github.com/dane-unltd/linalg/matrix"
)

//Armijo line search. Result placed in x.
func Armijo(op Problem, s float64, x, d Vec) (float64, float64) {
	beta := 0.5
	f0 := op.Obj.F(x)
	xTemp := NewVec(len(x))

	g := NewVec(len(x))
	op.Obj.G(x, g)
	gLin := Dot(g, d)

	xTemp.Copy(x)
	xTemp.Axpy(s, d)
	f := op.Obj.F(xTemp)

	if f-f0 > 0.5*gLin*s {
		fPrev := f
		s *= beta
		for {
			xTemp.Copy(x)
			xTemp.Axpy(s, d)
			f = op.Obj.F(xTemp)
			if f-f0 <= 0.5*gLin*s {
				if fPrev < f {
					s /= beta
					f = fPrev
				}
				break
			}
			fPrev = f
			s *= beta
		}
	} else {
		fPrev := f
		s /= beta
		for {
			xTemp.Copy(x)
			xTemp.Axpy(s, d)
			f = op.Obj.F(xTemp)
			if f-f0 > 0.5*gLin*s {
				if fPrev < f {
					s *= beta
					f = fPrev
				}
				break
			}
			fPrev = f
			s /= beta
		}
	}
	x.Axpy(s, d)
	return f, s
}

//extenden Armijo with projection
func ArmijoProj(op Problem, s float64, x, d Vec) (float64, float64) {
	beta := 0.5
	f0 := op.Obj.F(x)
	xTemp := NewVec(len(x))

	gApprox := NewVec(len(x))
	op.Obj.G(x, gApprox)
	gApprox.Scal(s)
	gApprox.Add(gApprox, x)
	op.Proj(gApprox)
	gApprox.Sub(x, gApprox)

	gLin := Dot(gApprox, d)

	xTemp.Copy(x)
	xTemp.Axpy(1e-3*s, d)
	op.Proj(xTemp)

	xTemp.Copy(x)
	xTemp.Axpy(s, d)
	op.Proj(xTemp)
	f := op.Obj.F(xTemp)

	if f-f0 > 0.5*gLin {
		fPrev := f
		s *= beta
		for {
			xTemp.Copy(x)
			xTemp.Axpy(s, d)
			op.Proj(xTemp)
			f = op.Obj.F(xTemp)
			if f-f0 <= 0.5*gLin {
				if fPrev < f {
					s /= beta
					f = fPrev
				}
				break
			}
			fPrev = f
			s *= beta
		}
	} else {
		fPrev := f
		s /= beta
		for {
			xTemp.Copy(x)
			xTemp.Axpy(s, d)
			op.Proj(xTemp)
			f = op.Obj.F(xTemp)
			if f-f0 > 0.5*gLin {
				if fPrev < f {
					s *= beta
					f = fPrev
				}
				break
			}
			fPrev = f
			s /= beta
		}
	}
	x.Axpy(s, d)
	if op.Proj != nil {
		op.Proj(x)
	}
	return f, s
}
