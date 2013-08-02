package linprog

import (
	. "github.com/dane-unltd/linalg/mat"
	"math"
)

//TODO: Predictor-Corrector Interior Point implementation
func linprog(c Vec, A *Dense, b Vec, tol float64) (x, y, s Vec) {
	m, n := A.Dims()

	At := A.TrView()

	var mu, sigma float64

	x = NewVec(n).AddSc(1)
	s = NewVec(n).AddSc(1)
	y = NewVec(m)

	dx := NewVec(n)
	ds := NewVec(n)
	dy := NewVec(m)

	dxAff := NewVec(n)
	dsAff := NewVec(n)
	dyAff := NewVec(m)

	dxCC := NewVec(n)
	dsCC := NewVec(n)
	dyCC := NewVec(m)

	rd := NewVec(n)
	rp := NewVec(m)
	rs := NewVec(n)

	xdivs := NewVec(n)
	temp := New(m, n)

	lhs := New(m, m)
	rhs := NewVec(m)
	soli := NewVec(m)

	triU := New(m, m)
	triUt := triU.TrView()

	nTemp1 := NewVec(n)
	nTemp2 := NewVec(n)

	alpha := 0.0

	for iter := 0; iter < 1000; iter++ {
		rd.Sub(c, s)
		rd.AddMul(At, y, -1)
		rp.Apply(A, x)
		rp.Sub(b, rp)
		rs.Mul(x, s)
		rs.Neg(rs)

		mu = rs.Asum() / float64(n)

		if (rd.Asum()+rp.Asum()+rs.Asum())/float64(n) < tol {
			break
		}

		//determining left hand side
		temp.ScalCols(A, xdivs.Div(x, s))
		lhs.Mul(temp, At)

		//factorization
		lhs.Chol(triU)

		//right hand side
		nTemp1.Add(rd, s)
		nTemp1.Mul(nTemp1, xdivs)
		rhs.Apply(A, nTemp1)
		rhs.Add(rhs, rp)

		//solving for dyAff
		soli.Trsv(triUt, rhs)
		dyAff.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		nTemp1.Apply(At, dyAff)
		dxAff.Sub(nTemp1, rd)
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
		mu_aff := Dot(nTemp1, nTemp2) / float64(n)

		//centering parameter
		sigma = math.Pow(mu_aff/mu, 3)

		//right hand side for predictor corrector step
		rs.Mul(dxAff, dsAff)
		rs.Neg(rs)
		rs.AddSc(sigma * mu_aff)

		nTemp1.Div(rs, s)
		nTemp1.Neg(nTemp1)

		rhs.Apply(A, nTemp1)

		//solving for dyCC
		soli.Trsv(triUt, rhs)
		dyCC.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		nTemp1.Apply(At, dyCC)
		dxCC.Mul(nTemp1, x)
		dxCC.Add(rs, dxCC)
		dxCC.Div(dxCC, s)

		dsCC.Mul(dxCC, s)
		dsCC.Sub(rs, dsCC)
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
	return
}
