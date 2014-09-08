package multi

import (
	"github.com/gonum/blas/dbw"

	"math"
)

type LineSearcher interface {
	Search(obj FdF, sol *Solution, dir []float64, step float64) float64
}

type Backtracking struct {
	Armijo float64
	xtmp   []float64
}

const (
	stepMin = 0.2
	stepMax = 0.8
)

func (b Backtracking) Search(obj FdF, sol *Solution, dir []float64, stepUpper float64) float64 {
	x := dbw.NewVector(sol.X)
	g := dbw.NewVector(sol.Grad)
	d := dbw.NewVector(dir)

	if b.xtmp == nil {
		b.xtmp = make([]float64, len(sol.X))
	}
	xtmp := dbw.NewVector(b.xtmp)

	gLin := dbw.Dot(g, d)

	f0 := sol.Obj

	dbw.Copy(x, xtmp)
	dbw.Axpy(stepUpper, d, xtmp)

	fUpper := obj.F(xtmp.Data)

	if fUpper-f0 < b.Armijo*gLin*stepUpper {
		dbw.Copy(xtmp, x)
		sol.Obj = fUpper
		obj.DF(sol.X, sol.Grad)
		return stepUpper
	}

	//quadratic interpolation
	step := stepUpper
	stepNew := -0.5 * gLin * stepUpper * stepUpper / (fUpper - f0 - gLin*stepUpper)
	if !(stepNew/step > stepMin) {
		step = stepMin * step
	} else if !(stepNew/step < stepMax) {
		step = stepMax * step
	} else {
		step = stepNew
	}

	dbw.Copy(x, xtmp)
	dbw.Axpy(step, d, xtmp)

	f := obj.F(xtmp.Data)

	for f-f0 > b.Armijo*gLin*step {
		stepSq := step * step
		stepUpperSq := stepUpper * stepUpper

		frac := 1 / (stepSq * stepUpperSq * (stepUpper - step))
		d1 := f - f0 - gLin*stepUpper
		d0 := fUpper - f0 - gLin*step

		a := frac * (stepSq*d1 - stepUpperSq*d0)
		b := frac * (stepUpperSq*stepUpper*d0 - stepSq*step*d1)

		stepUpper = step
		stepNew = (-b + math.Sqrt(b*b-3*a*gLin)) / (3 * a)
		if !(stepNew/step > stepMin) {
			step = stepMin * step
		} else if !(stepNew/step < stepMax) {
			step = stepMax * step
		} else {
			step = stepNew
		}

		fUpper = f

		dbw.Copy(x, xtmp)
		dbw.Axpy(step, d, xtmp)

		f = obj.F(xtmp.Data)

	}

	dbw.Copy(xtmp, x)
	sol.Obj = f
	obj.DF(sol.X, sol.Grad)

	return step
}
