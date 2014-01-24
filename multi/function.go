package multi

import (
	"github.com/gonum/blas/blasw"
)

type F interface {
	F(x []float64) float64
}

type FdF interface {
	F
	DF(x, g []float64)
	FdF(x, g []float64) float64
}

type FdFddF interface {
	FdF
	DDF(x, h []float64)
	FdFddF(x, g, h []float64) float64
}

type Projection interface {
	Project(x []float64)
}

type LineF struct {
	f           F
	x, d, xTemp blasw.Vector
	Dir         float64
}

func NewLineF(f F, x, d []float64) *LineF {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineF{
		f:     f,
		x:     blasw.NewVector(x),
		d:     blasw.NewVector(d),
		Dir:   1,
		xTemp: blasw.NewVector(make([]float64, n)),
	}
}

func (lf *LineF) F(alpha float64) float64 {
	blasw.Dcopy(lf.x, lf.xTemp)
	blasw.Daxpy(lf.Dir*alpha, lf.d, lf.xTemp)
	return lf.f.F(lf.xTemp.Data)
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
	x, d, g, xTemp blasw.Vector
}

func NewLineFdF(fdf FdF, x, d []float64) *LineFdF {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFdF{
		fdf:   fdf,
		x:     blasw.NewVector(x),
		d:     blasw.NewVector(d),
		g:     blasw.NewVector(make([]float64, n)),
		xTemp: blasw.NewVector(make([]float64, n)),
	}
}

func (lf *LineFdF) F(alpha float64) float64 {
	blasw.Dcopy(lf.x, lf.xTemp)
	blasw.Daxpy(alpha, lf.d, lf.xTemp)
	return lf.fdf.F(lf.xTemp.Data)
}

func (lf *LineFdF) DF(alpha float64) float64 {
	blasw.Dcopy(lf.x, lf.xTemp)
	blasw.Daxpy(alpha, lf.d, lf.xTemp)
	lf.fdf.DF(lf.xTemp.Data, lf.g.Data)

	return blasw.Ddot(lf.d, lf.g)
}

func (lf *LineFdF) FdF(alpha float64) (float64, float64) {
	blasw.Dcopy(lf.x, lf.xTemp)
	blasw.Daxpy(alpha, lf.d, lf.xTemp)
	val := lf.fdf.FdF(lf.xTemp.Data, lf.g.Data)

	return val, blasw.Ddot(lf.d, lf.g)
}

type LineFProj struct {
	f           F
	p           Projection
	x, d, xTemp blasw.Vector
}

func NewLineFProj(f F, p Projection, x, d []float64) *LineFProj {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFProj{
		f:     f,
		p:     p,
		x:     blasw.NewVector(x),
		d:     blasw.NewVector(d),
		xTemp: blasw.NewVector(make([]float64, n)),
	}
}

func (lf *LineFProj) F(alpha float64) float64 {
	blasw.Dcopy(lf.x, lf.xTemp)
	blasw.Daxpy(alpha, lf.d, lf.xTemp)
	lf.p.Project(lf.xTemp.Data)
	return lf.f.F(lf.xTemp.Data)
}
