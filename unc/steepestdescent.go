package unc

import (
	"github.com/dane-unltd/linalg/mat"
	"github.com/dane-unltd/opt"
	"github.com/dane-unltd/opt/linesearch"
)

type SteepestDescent struct {
	Tol        float64
	IterMax    int
	LineSearch linesearch.Solver
}

func (sd SteepestDescent) Solve(obj opt.Miso, grad opt.Mimo, x mat.Vec) opt.Result {
	s := 1.0
	fHist := make([]float64, 0)
	f := obj(x)
	fHist = append(fHist, f)
	d := mat.NewVec(len(x))
	gLin := 0.0

	xTemp := mat.NewVec(len(x))

	lineFun := func(s float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(s, d)
		return obj(xTemp)
	}

	m := linesearch.Model{F: lineFun}

	i := 0
	for ; i < sd.IterMax; i++ {
		grad(x, d)
		d.Scal(-1)

		gLin = -d.Nrm2Sq()

		if gLin/float64(len(x)) > -sd.Tol {
			break
		}

		m.LBF, m.LBG, m.X = f, gLin, s
		res, _ := sd.LineSearch.Solve(&m)
		s, f = res.X, res.F

		fHist = append(fHist, f)

		x.Axpy(s, d)
	}
	res := opt.Result{
		Obj:     f,
		Iter:    i,
		Grad:    d.Scal(-1),
		ObjHist: fHist,
	}
	if i == sd.IterMax {
		res.Status = opt.MaxIter
	}
	return res
}
