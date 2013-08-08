package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type Function interface {
	Val(x mat.Vec) float64
}

type Grad interface {
	Function
	ValGrad(x, g mat.Vec) float64
}

type Hessian interface {
	Grad
	ValGradHess(x, g, h mat.Vec) float64
}

type Projection interface {
	Project(x mat.Vec)
}

type LineFunc struct {
	f           Function
	x, d, xTemp mat.Vec
	Dir         float64
}

func NewLineFunc(f Function, x, d mat.Vec) *LineFunc {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFunc{
		f:     f,
		x:     x,
		d:     d,
		Dir:   1,
		xTemp: mat.NewVec(n),
	}
}

func (lf *LineFunc) Val(alpha float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(lf.Dir*alpha, lf.d)
	return lf.f.Val(lf.xTemp)
}

func (lf *LineFunc) SwitchDir() {
	if lf.Dir > 0 {
		lf.Dir = -1
	} else {
		lf.Dir = 1
	}
}

type LineFuncDeriv struct {
	f              Grad
	x, d, g, xTemp mat.Vec
}

func NewLineFuncDeriv(f Grad, x, d mat.Vec) *LineFuncDeriv {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFuncDeriv{
		f:     f,
		x:     x,
		d:     d,
		g:     mat.NewVec(n),
		xTemp: mat.NewVec(n),
	}
}

func (lf *LineFuncDeriv) Val(x float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	return lf.f.Val(lf.xTemp)
}

func (lf *LineFuncDeriv) ValDeriv(x float64) (float64, float64) {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	val := lf.f.ValGrad(lf.xTemp, lf.g)

	return val, mat.Dot(lf.d, lf.g)
}

type LineFuncProj struct {
	f           Function
	p           Projection
	x, d, xTemp mat.Vec
}

func NewLineFuncProj(f Function, p Projection, x, d mat.Vec) *LineFuncProj {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFuncProj{
		f:     f,
		p:     p,
		x:     x,
		d:     d,
		xTemp: mat.NewVec(n),
	}
}

func (lf *LineFuncProj) Val(x float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	lf.p.Project(lf.xTemp)
	return lf.f.Val(lf.xTemp)
}
