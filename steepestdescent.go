package opt

import (
	"github.com/dane-unltd/linalg/matrix"
	"github.com/dane-unltd/opt/linesearch"
)

type Method int

const (
	Exact Method = iota
	Inexact
)

func SteepestDescent(obj Objective, x matrix.Vec,
	zeta float64, nIter int, m Method) (float64, int) {
	s := 1.0
	f := obj.F(x)
	d := matrix.NewVec(len(x))
	gLin := 0.0

	xTemp := matrix.NewVec(len(x))

	lineFun := func(s float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(s, d)
		return obj.F(xTemp)
	}
	i := 0
	for ; i < nIter; i++ {
		obj.G(x, d)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()

		if gLin/float64(len(x)) > -zeta {
			break
		}

		if m == Inexact {
			s, f = linesearch.Inexact(lineFun, f, gLin, s)
		} else if m == Exact {
			s, f = linesearch.Exact(lineFun, f, s, 0.1*s)
		} else {
			panic("unknown line search method")
		}
		x.Axpy(s, d)
	}
	return f, i
}
