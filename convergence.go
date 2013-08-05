package opt

import (
	"github.com/dane-unltd/linalg/mat"
	"math"
	"time"
)

type GradXer interface {
	GradX() mat.Vec
}

type ObjXer interface {
	ObjX() float64
}

type Xer interface {
	X() mat.Vec
}

type GradConv struct {
	TolAbsGrad float64
	TolRelGrad float64
	g0         float64
}

type ObjConv struct {
	TolAbsObj float64
	TolRelObj float64
	prevObj   float64
}

type DeltaXConv struct {
	TolDeltaX float64
	prevX     mat.Vec
}

type NonConv struct {
	IterMax int
	TimeMax time.Duration
}

func (C GradConv) Init(m GradXer) {
	C.g0 = m.GradX().Nrm2()
}

func (C GradConv) Check(m GradXer) bool {
	normGrad := m.GradX().Nrm2()
	if normGrad < C.TolAbsGrad {
		return true
	}
	if normGrad/C.g0 < C.TolRelGrad {
		return true
	}
	return false
}

func (C ObjConv) Check(m ObjXer) bool {
	currObj := m.ObjX()
	if C.prevObj == 0 {
		C.prevObj = currObj
		return false
	}
	if C.prevObj-currObj > -C.TolAbsObj {
		C.prevObj = currObj
		return true
	}
	if (C.prevObj-currObj)/math.Abs(currObj) > -C.TolRelObj {
		C.prevObj = currObj
		return true
	}
	C.prevObj = currObj
	return false
}
