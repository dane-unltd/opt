package linprog

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

type PredCorr struct {
}

func NewPredCorr() *PredCorr {
	return &PredCorr{}
}

//Predictor-Corrector Interior Point implementation
func (sol *PredCorr) Solve(mdl *Model) Status {
	var status Status

	A := mdl.A

	m, n := A.Dims()

	At := A.TrView()

	var mu, sigma float64

	mdl.X = mat.NewVec(n).AddSc(1)
	mdl.S = mat.NewVec(n).AddSc(1)
	mdl.Y = mat.NewVec(m)
	x := mdl.X
	s := mdl.S
	y := mdl.Y

	dx := mat.NewVec(n)
	ds := mat.NewVec(n)
	dy := mat.NewVec(m)

	dxAff := mat.NewVec(n)
	dsAff := mat.NewVec(n)
	dyAff := mat.NewVec(m)

	dxCC := mat.NewVec(n)
	dsCC := mat.NewVec(n)
	dyCC := mat.NewVec(m)

	xdivs := mat.NewVec(n)
	temp := mat.New(m, n)

	lhs := mat.New(m, m)
	rhs := mat.NewVec(m)
	soli := mat.NewVec(m)

	triU := mat.New(m, m)
	triUt := triU.TrView()

	nTemp1 := mat.NewVec(n)
	nTemp2 := mat.NewVec(n)

	alpha := 0.0

	mdl.init()
	for {

		if status = mdl.update(); status != 0 {
			break
		}
		mu = mdl.rs.Asum() / float64(n)

		//determining left hand side
		temp.ScalCols(A, xdivs.Div(x, s))
		lhs.Mul(temp, At)

		//factorization
		lhs.Chol(triU)

		//right hand side
		nTemp1.Add(mdl.rd, s)
		nTemp1.Mul(nTemp1, xdivs)
		rhs.Apply(A, nTemp1)
		rhs.Add(rhs, mdl.rp)

		//solving for dyAff
		soli.Trsv(triUt, rhs)
		dyAff.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		nTemp1.Apply(At, dyAff)
		dxAff.Sub(nTemp1, mdl.rd)
		dxAff.Sub(dxAff, s)
		dxAff.Mul(dxAff, xdivs)

		dsAff.Div(dxAff, xdivs)
		dsAff.Add(dsAff, s)
		dsAff.Neg(dsAff)

		//determining step size
		alpha = 1.0
		for i := range dxAff {
			if dxAff[i] < 0 {
				alph := -x[i] / dxAff[i]
				if alph < alpha {
					alpha = alph
				}
			}
		}
		for i := range dsAff {
			if dsAff[i] < 0 {
				alph := -s[i] / dsAff[i]
				if alph < alpha {
					alpha = alph
				}
			}
		}
		alpha *= 0.99995

		//calculating duality gap measure for affine case
		nTemp1.Copy(x)
		nTemp1.Axpy(alpha, dxAff)
		nTemp2.Copy(s)
		nTemp2.Axpy(alpha, dsAff)
		mu_aff := mat.Dot(nTemp1, nTemp2) / float64(n)

		//centering parameter
		sigma = math.Pow(mu_aff/mu, 3)

		//right hand side for predictor corrector step
		mdl.rs.Mul(dxAff, dsAff)
		mdl.rs.Neg(mdl.rs)
		mdl.rs.AddSc(sigma * mu_aff)

		nTemp1.Div(mdl.rs, s)
		nTemp1.Neg(nTemp1)

		rhs.Apply(A, nTemp1)

		//solving for dyCC
		soli.Trsv(triUt, rhs)
		dyCC.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		nTemp1.Apply(At, dyCC)
		dxCC.Mul(nTemp1, x)
		dxCC.Add(mdl.rs, dxCC)
		dxCC.Div(dxCC, s)

		dsCC.Mul(dxCC, s)
		dsCC.Sub(mdl.rs, dsCC)
		dsCC.Div(dsCC, x)

		dx.Add(dxAff, dxCC)
		dy.Add(dyAff, dyCC)
		ds.Add(dsAff, dsCC)

		alpha = 1
		for i := range dx {
			if dx[i] < 0 {
				alph := -x[i] / dx[i]
				if alph < alpha {
					alpha = alph
				}
			}
		}
		for i := range ds {
			if ds[i] < 0 {
				alph := -s[i] / ds[i]
				if alph < alpha {
					alpha = alph
				}
			}
		}
		alpha *= 0.99995

		x.Axpy(alpha, dx)
		y.Axpy(alpha, dy)
		s.Axpy(alpha, ds)
	}

	return status
}
