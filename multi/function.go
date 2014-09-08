package multi

import (
	"github.com/gonum/blas/dbw"
)

type FdF interface {
	F(x []float64) float64
	DF(x, g []float64)
	FdF(x, g []float64) float64
}

type LineFdF struct {
	fdf            FdF
	x, d, g, xTemp dbw.Vector
}

func NewLineFdF(fdf FdF, x, d []float64) *LineFdF {
	n := len(x)
	if len(d) != n {
		panic("dimension mismatch")
	}
	return &LineFdF{
		fdf:   fdf,
		x:     dbw.NewVector(x),
		d:     dbw.NewVector(d),
		g:     dbw.NewVector(make([]float64, n)),
		xTemp: dbw.NewVector(make([]float64, n)),
	}
}

func (lf *LineFdF) F(alpha float64) float64 {
	dbw.Copy(lf.x, lf.xTemp)
	dbw.Axpy(alpha, lf.d, lf.xTemp)
	return lf.fdf.F(lf.xTemp.Data)
}

func (lf *LineFdF) DF(alpha float64) float64 {
	dbw.Copy(lf.x, lf.xTemp)
	dbw.Axpy(alpha, lf.d, lf.xTemp)
	lf.fdf.DF(lf.xTemp.Data, lf.g.Data)

	return dbw.Dot(lf.d, lf.g)
}

func (lf *LineFdF) FdF(alpha float64) (float64, float64) {
	dbw.Copy(lf.x, lf.xTemp)
	dbw.Axpy(alpha, lf.d, lf.xTemp)
	val := lf.fdf.FdF(lf.xTemp.Data, lf.g.Data)

	return val, dbw.Dot(lf.d, lf.g)
}

type Wrapper struct {
	Stats *Stats
	Func  FdF
}

func (w Wrapper) F(x []float64) float64 {
	w.Stats.FunEvals++
	return w.Func.F(x)
}

func (w Wrapper) DF(x, g []float64) {
	w.Stats.GradEvals++
	w.Func.DF(x, g)
}

func (w Wrapper) FdF(x, g []float64) float64 {
	w.Stats.FunEvals++
	w.Stats.GradEvals++
	return w.Func.FdF(x, g)
}
