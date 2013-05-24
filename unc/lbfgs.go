package unc

import (
	"github.com/dane-unltd/linalg/matrix"
	"github.com/dane-unltd/opt/linesearch"
)

type LBFGSSolver struct {
	Tol        float64
	IterMax    int
	Mem        int
	LineSearch linesearch.Solver
}

func (sol LBFGSSolver) Solve(obj Miso, grad Mimo, x matrix.Vec) Result {
	stepSize := 1.0
	n := len(x)

	fHist := make([]float64, 0)
	f := obj(x)
	fHist = append(fHist, f)
	gLin := 0.0

	S := make([]matrix.Vec, sol.Mem)
	Y := make([]matrix.Vec, sol.Mem)
	for i := 0; i < sol.Mem; i++ {
		S[i] = matrix.NewVec(n)
		Y[i] = matrix.NewVec(n)
	}

	d := matrix.NewVec(n)
	g := matrix.NewVec(n)

	xOld := matrix.NewVec(n)
	gOld := matrix.NewVec(n)
	sNew := matrix.NewVec(n)
	yNew := matrix.NewVec(n)

	alphas := matrix.NewVec(sol.Mem)
	betas := matrix.NewVec(sol.Mem)
	rhos := matrix.NewVec(sol.Mem)

	xTemp := matrix.NewVec(n)
	lineFun := func(step float64) float64 {
		xTemp.Copy(x)
		xTemp.Axpy(step, d)
		return obj(xTemp)
	}

	iter := 0
	for ; iter < sol.IterMax; iter++ {
		grad(x, g)
		d.Copy(g)

		if iter > 0 {
			yNew.Sub(g, gOld)
			sNew.Sub(x, xOld)

			temp := S[len(S)-1]
			copy(S[1:], S)
			S[0] = temp
			S[0].Copy(sNew)

			temp = Y[len(S)-1]
			copy(Y[1:], Y)
			Y[0] = temp
			Y[0].Copy(yNew)

			copy(rhos[1:], rhos)
			rhos[0] = 1 / matrix.Dot(sNew, yNew)
			for i := 0; i < sol.Mem; i++ {
				alphas[i] = rhos[i] * matrix.Dot(S[i], d)
				d.Axpy(-alphas[i], Y[i])
			}
			for i := sol.Mem - 1; i >= 0; i-- {
				betas[i] = rhos[i] * matrix.Dot(Y[i], d)
				d.Axpy(alphas[i]-betas[i], S[i])
			}
		}

		d.Scal(-1)

		gLin = matrix.Dot(d, g)

		if gLin/float64(len(x)) > -sol.Tol {
			break
		}

		stepSize, f = sol.LineSearch.Solve(lineFun, nil, f, gLin,
			stepSize)

		xOld.Copy(x)
		gOld.Copy(g)

		fHist = append(fHist, f)

		x.Axpy(stepSize, d)
	}
	res := Result{
		Obj:     f,
		Iter:    iter,
		Grad:    g,
		ObjHist: fHist,
	}
	if iter == sol.IterMax {
		res.Status = MaxIter
	}
	return res
}
