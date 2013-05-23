package opt

import . "github.com/dane-unltd/linalg/matrix"

func linprog(c Vec, A *Dense, b Vec) {
	m, n := A.Size()

	At := A.TrView()

	var mu, sigma float64

	x := NewVec(n).AddSc(1)
	s := NewVec(n).AddSc(1)
	y := NewVec(m)

	dx := NewVec(n)
	ds := NewVec(n)
	dy := NewVec(m)

	rd := NewVec(n)
	rp := NewVec(m)
	rs := NewVec(n)

	sinvx := NewVec(n)
	temp := NewDense(m, n)
	lhs := NewDense(m, m)
	rhs := NewVec(m)
	soli := NewVec(m)
	triU := NewDense(m, m)
	triUt := triU.TrView()
	rhstemp := NewVec(n)

	alpha := 0.0

	for iter := 0; iter < 10; iter++ {

		rd.Sub(s, c)
		rd.AddMul(At, y, -1)
		rp.Mul(A, x)
		rp.Sub(b, rp)
		rs.MulH(x, s)
		rs.Neg(rs)

		temp.Mul(A, Diag(sinvx.DivH(x, s)))
		lhs.Mul(temp, At)
		lhs.Chol(triU)

		rhs.Neg(rp)
		rhstemp.DivH(rs, x)
		rhstemp.Add(rhstemp, rd)
		rhs.AddMul(temp, rhstemp, 1)

		soli.Trsv(triUt, rhs)
		dy.Trsv(triU, soli)

		dx.Mul(At, dy)
		rhstemp.DivH(rs, x)
		dx.Add(dx, rhstemp)
		dx.Add(dx, rd)
		dx.MulH(dx, x)
		dx.DivH(dx, s)

		ds.DivH(rs, x)
		rhstemp.MulH(dx, s)
		rhstemp.DivH(rhstemp, x)
		ds.Sub(ds, rhstemp)

		rhstemp.DivH(x, dx)

		_ = mu
		_ = sigma
		_ = alpha
	}
}
