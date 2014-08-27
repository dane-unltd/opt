package multi

import (
	"github.com/gonum/blas/dbw"
)

type LBFGS struct {
	S, Y []dbw.Vector

	xOld, gOld dbw.Vector
	sNew, yNew dbw.Vector

	alphas, betas, rhos []float64

	notFirst bool
	Mem      int
}

func (sol *LBFGS) SearchDirection(s Solution, dSl []float64) {
	n := len(dSl)
	if !sol.notFirst {
		if sol.Mem <= 0 {
			sol.Mem = 5
		}
		*sol = LBFGS{
			S: make([]dbw.Vector, sol.Mem),
			Y: make([]dbw.Vector, sol.Mem),

			xOld: dbw.NewVector(make([]float64, n)),
			gOld: dbw.NewVector(make([]float64, n)),
			sNew: dbw.NewVector(make([]float64, n)),
			yNew: dbw.NewVector(make([]float64, n)),

			alphas: make([]float64, sol.Mem),
			betas:  make([]float64, sol.Mem),
			rhos:   make([]float64, sol.Mem),

			Mem: sol.Mem,
		}
		for i := 0; i < sol.Mem; i++ {
			sol.S[i] = dbw.NewVector(make([]float64, n))
			sol.Y[i] = dbw.NewVector(make([]float64, n))
		}
	}

	d := dbw.NewVector(dSl)
	g := dbw.NewVector(s.Grad)
	x := dbw.NewVector(s.X)

	dbw.Copy(g, d)

	if sol.notFirst {
		dbw.Copy(g, sol.yNew)
		dbw.Axpy(-1, sol.gOld, sol.yNew)
		dbw.Copy(x, sol.sNew)
		dbw.Axpy(-1, sol.xOld, sol.sNew)

		temp := sol.S[len(sol.S)-1]
		copy(sol.S[1:], sol.S)
		sol.S[0] = temp
		dbw.Copy(sol.sNew, sol.S[0])

		temp = sol.Y[len(sol.S)-1]
		copy(sol.Y[1:], sol.Y)
		sol.Y[0] = temp
		dbw.Copy(sol.yNew, sol.Y[0])

		copy(sol.rhos[1:], sol.rhos)
		sol.rhos[0] = 1 / dbw.Dot(sol.sNew, sol.yNew)
		for i := 0; i < len(sol.alphas); i++ {
			sol.alphas[i] = sol.rhos[i] * dbw.Dot(sol.S[i], d)
			dbw.Axpy(-sol.alphas[i], sol.Y[i], d)
		}
		for i := len(sol.alphas) - 1; i >= 0; i-- {
			sol.betas[i] = sol.rhos[i] * dbw.Dot(sol.Y[i], d)
			dbw.Axpy(sol.alphas[i]-sol.betas[i], sol.S[i], d)
		}
	}
	sol.notFirst = true

	dbw.Scal(-1, d)
	dbw.Copy(x, sol.xOld)
	dbw.Copy(g, sol.gOld)
}
