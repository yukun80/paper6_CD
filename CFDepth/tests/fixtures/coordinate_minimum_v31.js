'use strict';
// v3.1数学内核，作为优化前的固定回归/构图成本基线。
function coordinateMinimum(v,o) {
  var z=o.c(0), big=o.c(1e12), best=o.c(1e30), answer=z;
  var square=function(x){return o.mul(x,x);};
  for (var side=-1;side<=1;side++) {
    for (var below=0;below<=1;below++) {
      var low=side===1?v.upper:o.sub(z,big), high=side===-1?v.lower:big;
      if (side===0) {low=v.lower;high=v.upper;}
      low=o.max(low,below?o.sub(z,big):v.terrain);
      high=o.min(high,below?v.terrain:big);
      low=o.max(low,o.choose(v.hard,v.terrain,o.sub(z,big)));
      var bw=side===0?z:v.boundary, target=side===-1?v.lower:v.upper, tw=below?v.soft:z;
      var den=o.add(o.add(v.degree,bw),o.add(tw,v.midWeight));
      var num=o.add(o.add(v.neighborSum,o.mul(bw,target)),
        o.add(o.mul(tw,v.terrain),o.mul(v.midWeight,v.mid)));
      var s=o.max(low,o.min(high,o.div(num,o.max(den,o.c(1e-30)))));
      var mean=o.div(v.neighborSum,o.max(v.degree,o.c(1e-30)));
      var dist=o.max(z,o.max(o.sub(v.lower,s),o.sub(s,v.upper)));
      var energy=o.add(o.mul(v.degree,square(o.sub(s,mean))),
        o.add(o.mul(v.boundary,square(dist)),o.add(o.mul(v.soft,square(o.max(z,o.sub(v.terrain,s)))),
          o.mul(v.midWeight,square(o.sub(s,v.mid))))));
      var take=o.and(o.lte(low,high),o.lt(energy,best));
      answer=o.choose(take,s,answer);best=o.choose(take,energy,best);
    }
  }
  return answer;
}

module.exports=coordinateMinimum;
