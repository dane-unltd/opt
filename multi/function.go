package multi

import (
	"github.com/dane-unltd/linalg/mat"
)

type F interface {
	F(x mat.Vec) float64
}

type FdF interface {
	F
	DF(x, g mat.Vec)
	FdF(x, g mat.Vec) float64
}

type FdFddF interface {
	FdF
	DDF(x, h mat.Vec)
	FdFddF(x, g, h mat.Vec) float64
}

type Projection interface {
	Project(x mat.Vec)
}

type LineF struct {
	f           F
	x, d, xTemp mat.Vec
	Dir         float64
}

func NewLineF(f F, x, d mat.Vec) *LineF {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineF{
		f:     f,
		x:     x,
		d:     d,
		Dir:   1,
		xTemp: mat.NewVec(n),
	}
}

func (lf *LineF) F(alpha float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(lf.Dir*alpha, lf.d)
	return lf.f.F(lf.xTemp)
}

func (lf *LineF) SwitchDir() {
	if lf.Dir > 0 {
		lf.Dir = -1
	} else {
		lf.Dir = 1
	}
}

type LineFdF struct {
	fdf            FdF
	x, d, g, xTemp mat.Vec
}

func NewLineFdF(fdf FdF, x, d mat.Vec) *LineFdF {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFdF{
		fdf:   fdf,
		x:     x,
		d:     d,
		g:     mat.NewVec(n),
		xTemp: mat.NewVec(n),
	}
}

func (lf *LineFdF) F(x float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	return lf.fdf.F(lf.xTemp)
}

func (lf *LineFdF) DF(x float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	lf.fdf.DF(lf.xTemp, lf.g)

	return mat.Dot(lf.d, lf.g)
}

func (lf *LineFdF) FdF(x float64) (float64, float64) {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	val := lf.fdf.FdF(lf.xTemp, lf.g)

	return val, mat.Dot(lf.d, lf.g)
}

type LineFProj struct {
	f           F
	p           Projection
	x, d, xTemp mat.Vec
}

func NewLineFProj(f F, p Projection, x, d mat.Vec) *LineFProj {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFProj{
		f:     f,
		p:     p,
		x:     x,
		d:     d,
		xTemp: mat.NewVec(n),
	}
}

func (lf *LineFProj) F(x float64) float64 {
	lf.xTemp.Copy(lf.x)
	lf.xTemp.Axpy(x, lf.d)
	lf.p.Project(lf.xTemp)
	return lf.f.F(lf.xTemp)
}
