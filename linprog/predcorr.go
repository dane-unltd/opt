package linprog

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
)

type PredCorr struct{}

func NewPredCorr() *PredCorr {
	return &PredCorr{}
}

func checkKKT(r *Result, p *Params) Status {
	if r.Rd.Asum() < p.Infeasibility &&
		r.Rp.Asum() < p.Infeasibility &&
		r.Rs.Asum() < p.DualityGap {
		r.Status = Success
		return r.Status
	}
	return r.Status
}

//Predictor-Corrector Interior Point implementation
func (sol *PredCorr) Solve(prob *Problem, p *Params, upd ...Updater) *Result {
	res := NewResult(prob)
	upd = append(upd, newBasicConv(p))

	A := prob.A

	m, n := A.Dims()

	At := A.TrView()

	var mu, sigma float64

	//parameter for step size scaling
	gamma := 0.01

	res.X.AddSc(1)
	res.S.AddSc(1)

	res.Rd = mat.NewVec(n)
	res.Rp = mat.NewVec(m)
	res.Rs = mat.NewVec(n)

	x := res.X
	s := res.S
	y := res.Y

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

	alphaPrimal := 1.0
	alphaDual := 1.0

	for {
		res.Rd.Sub(prob.C, res.S)
		res.Rd.AddMul(At, res.Y, -1)
		res.Rp.Transform(A, res.X)
		res.Rp.Sub(prob.B, res.Rp)
		res.Rs.Mul(res.X, res.S)
		res.Rs.Neg(res.Rs)

		if doUpdates(res, upd) != 0 {
			break
		}
		if checkKKT(res, p); res.Status != 0 {
			break
		}
		if mat.Dot(prob.C, x) < -1e10 {
			res.Status = Unbounded
			break
		}
		if mat.Dot(y, prob.B) > 1e10 || alphaPrimal < 1e-4 || alphaDual < 1e-4 {
			res.Status = Infeasible
			break
		}

		mu = res.Rs.Asum() / float64(n)

		//determining left hand side
		temp.ScalCols(A, xdivs.Div(x, s))
		lhs.Mul(temp, At)

		//factorization
		info := lhs.Chol(triU)
		if info > 0 {
			res.Status = Fail
			break
		}

		//right hand side
		nTemp1.Add(res.Rd, s)
		nTemp1.Mul(nTemp1, xdivs)
		rhs.Transform(A, nTemp1)
		rhs.Add(rhs, res.Rp)

		//solving for dyAff
		soli.Trsv(triUt, rhs)
		dyAff.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		nTemp1.Transform(At, dyAff)
		dxAff.Sub(nTemp1, res.Rd)
		dxAff.Sub(dxAff, s)
		dxAff.Mul(dxAff, xdivs)

		dsAff.Div(dxAff, xdivs)
		dsAff.Add(dsAff, s)
		dsAff.Neg(dsAff)

		//determining step size
		alphaPrimal, _ = maxStep(x, dxAff)
		alphaDual, _ = maxStep(s, dsAff)

		//calculating duality gap measure for affine case
		nTemp1.Copy(x)
		nTemp1.Axpy(alphaPrimal, dxAff)
		nTemp2.Copy(s)
		nTemp2.Axpy(alphaDual, dsAff)
		mu_aff := mat.Dot(nTemp1, nTemp2) / float64(n)

		//centering parameter
		sigma = math.Pow(mu_aff/mu, 3)

		//right hand side for predictor corrector step
		res.Rs.Mul(dxAff, dsAff)
		res.Rs.Neg(res.Rs)
		res.Rs.AddSc(sigma * mu_aff)

		nTemp1.Div(res.Rs, s)
		nTemp1.Neg(nTemp1)

		rhs.Transform(A, nTemp1)

		//solving for dyCC
		soli.Trsv(triUt, rhs)
		dyCC.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		nTemp1.Transform(At, dyCC)
		dxCC.Mul(nTemp1, x)
		dxCC.Add(res.Rs, dxCC)
		dxCC.Div(dxCC, s)

		dsCC.Mul(dxCC, s)
		dsCC.Sub(res.Rs, dsCC)
		dsCC.Div(dsCC, x)

		dx.Add(dxAff, dxCC)
		dy.Add(dyAff, dyCC)
		ds.Add(dsAff, dsCC)

		//determining step size
		alphaPrimalMax, ixPrimal := maxStep(x, dx)
		alphaDualMax, ixDual := maxStep(s, ds)

		//calculating duality gap measure with full step length
		nTemp1.Copy(x)
		nTemp1.Axpy(alphaPrimalMax, dx)
		nTemp2.Copy(s)
		nTemp2.Axpy(alphaDualMax, ds)
		mu_f := mat.Dot(nTemp1, nTemp2) / float64(n)
		nTemp1.Mul(nTemp1, nTemp2)

		//step length calculations
		alphaPrimal = 1
		if ixPrimal >= 0 {
			fPrimal := (gamma*mu_f/(s[ixPrimal]+alphaDualMax*ds[ixPrimal]) - x[ixPrimal]) / (alphaPrimalMax * dx[ixPrimal])
			alphaPrimal = math.Max(1-gamma, fPrimal) * alphaPrimalMax
		}
		alphaDual = 1
		if ixDual >= 0 {
			fDual := (gamma*mu_f/(x[ixDual]+alphaPrimalMax*dx[ixDual]) - s[ixDual]) / (alphaDualMax * ds[ixDual])
			alphaDual = math.Max(1-gamma, fDual) * alphaDualMax
		}

		x.Axpy(alphaPrimal, dx)
		y.Axpy(alphaDual, dy)
		s.Axpy(alphaDual, ds)
	}
	return res
}

//Maximum step-size in [0, 1] such that all elements stay positive
func maxStep(x, dx mat.Vec) (alphaMax float64, ix int) {
	alphaMax = 1.0
	ix = -1
	for i, d := range dx {
		if d < 0 {
			alph := -x[i] / d
			if alph < alphaMax {
				alphaMax = alph
				ix = i
			}
		}
	}
	return
}
