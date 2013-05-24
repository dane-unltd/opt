package opt

import (
	"fmt"
	. "github.com/dane-unltd/linalg/matrix"
	"math"
)

//TODO: Predictor-Corrector Interior Point implementation
func linprog(c Vec, A *Dense, b Vec) Vec {
	m, n := A.Size()

	At := A.TrView()

	var mu, sigma float64

	x := NewVec(n).AddSc(1)
	s := NewVec(n).AddSc(1)
	y := NewVec(m)

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
	temp := NewDense(m, n)

	lhs := NewDense(m, m)
	rhs := NewVec(m)
	soli := NewVec(m)

	triU := NewDense(m, m)
	triUt := triU.TrView()

	rhstemp := NewVec(n)
	nTemp1 := NewVec(n)
	nTemp2 := NewVec(n)

	alpha := 0.0

	for iter := 0; iter < 50; iter++ {

		rd.Sub(s, c)
		rd.AddMul(At, y, -1)
		rp.Mul(A, x)
		rp.Sub(b, rp)
		rs.MulH(x, s)
		rs.Neg(rs)

		mu = rs.Asum()

		fmt.Println("rhs", mu)
		fmt.Println(rd, rp, rs)
		fmt.Println("x", x)

		//determining left hand side
		temp.Mul(A, Diag(xdivs.DivH(x, s)))
		lhs.Mul(temp, At)

		//factorization
		lhs.Chol(triU)

		//right hand side
		rhs.Neg(rp)
		rhstemp.DivH(rs, x)
		rhstemp.Add(rhstemp, rd)
		rhs.AddMul(temp, rhstemp, 1)

		//solving for dyAff
		soli.Trsv(triUt, rhs)
		dyAff.Trsv(triU, soli)

		//calculating other steps (dxAff, dsAff)
		dxAff.Mul(At, dyAff)
		nTemp1.DivH(rs, x)
		dxAff.Add(dxAff, nTemp1)
		dxAff.Add(dxAff, rd)
		dxAff.MulH(dxAff, x)
		dxAff.DivH(dxAff, s)

		dsAff.DivH(rs, x)
		nTemp1.MulH(dxAff, s)
		nTemp1.DivH(nTemp1, x)
		dsAff.Sub(dsAff, nTemp1)

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

		//calculating duality gap measure for affine case
		nTemp1.Copy(x)
		nTemp1.Axpy(alpha, dxAff)
		nTemp2.Copy(s)
		nTemp2.Axpy(alpha, dsAff)
		mu_aff := Dot(nTemp1, nTemp2) / float64(n)

		//centering parameter
		sigma = math.Pow(mu_aff/mu, 3)

		//right hand side for predictor corrector step
		rhstemp.MulH(dxAff, dsAff)
		rhstemp.Neg(rhstemp)
		rhstemp.AddSc(sigma * mu_aff)
		nTemp1.DivH(rhstemp, s)
		rhs.Mul(A, nTemp1)
		rhs.Neg(rhs)

		soli.Trsv(triUt, rhs)
		dyCC.Trsv(triU, soli)
		dxCC.Mul(At, dyCC)
		dxCC.MulH(dxCC, x).Add(rhstemp, dxCC)
		dxCC.DivH(dxCC, s)
		nTemp1.MulH(s, dxCC)
		dsCC.Sub(rhstemp, nTemp1).DivH(dsCC, x)

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
		alpha *= 0.995
		x.Axpy(alpha, dx)
		y.Axpy(alpha, dy)
		s.Axpy(alpha, ds)

	}
	return x
}
