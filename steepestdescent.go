package opt

import (
	. "github.com/dane-unltd/linalg/matrix"
)

func SteepestDescent(op Problem, x Vec, zeta float64, nIter int) float64 {
	s := 1.0
	f := 0.0
	d := NewVec(len(x))
	g := NewVec(len(x))
	gLin := 0.0
	for i := 0; i < nIter; i++ {
		op.Obj.G(x, g)
		d.Copy(g)
		d.Scal(-1)

		if op.Proj != nil {
			gApprox := NewVec(len(x))
			gApprox.Copy(g)
			gApprox.Scal(s)
			gApprox.Add(gApprox, x)
			op.Proj(gApprox)
			gApprox.Sub(x, gApprox)

			gLin = Dot(gApprox, d) / s
		} else {
			gLin = Dot(g, d)
		}
		if gLin/float64(len(x)) > -zeta {
			break
		}

		if op.Proj != nil {
			f, s = ArmijoProj(op, s, x, d)
		} else {
			f, s = Armijo(op, s, x, d)
		}
	}
	return f
}
