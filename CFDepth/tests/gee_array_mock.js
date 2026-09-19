'use strict';
// 小网格运算适配器用于实际生产sweep/energy路径；不是GEE服务端或资产模拟。
module.exports=function makeGrid(width,height,x0=0,y0=0){
  const n=width*height,ones=()=>Array(n).fill(1);
  const coord=(i)=>[i%width+x0,Math.floor(i/width)+y0];
  class Img {
    constructor(bands){this.bands=bands;}
    static one(values,name='constant',mask=ones()){return new Img({[name]:{v:values.slice(),m:mask.slice()}});}
    first(){return this.bands[Object.keys(this.bands)[0]];}
    reproject(){return this;}setDefaultProjection(){return this;}clip(){return this;}
    toDouble(){return this;}toByte(){return this;}
    toInt64(){return this.mapValues(Math.trunc);}floor(){return this.mapValues(Math.floor);}
    mapValues(fn){return new Img(Object.fromEntries(Object.entries(this.bands).map(([k,b])=>[k,{v:b.v.map(fn),m:b.m.slice()}])));}
    select(names){if(typeof names==='string')names=[names];const out={};for(const name of names){if(!this.bands[name])throw Error('Missing selected band '+name);out[name]=this.bands[name];}return new Img(out);}
    rename(names){if(typeof names==='string')names=[names];const old=Object.values(this.bands);if(names.length!==old.length)throw Error('Rename arity');return new Img(Object.fromEntries(names.map((name,i)=>[name,old[i]])));}
    addBands(other,unused,overwrite=false){const out={...this.bands};for(const [k,b]of Object.entries(other.bands)){if(out[k]&&!overwrite)throw Error('Duplicate band '+k);out[k]=b;}return new Img(out);}
    binary(other,fn){
      if(!(other instanceof Img))other=Img.one(Array(n).fill(other));
      const ak=Object.keys(this.bands),bk=Object.keys(other.bands),count=Math.max(ak.length,bk.length),out={};
      if(ak.length!==bk.length&&ak.length!==1&&bk.length!==1)throw Error('Band arity mismatch');
      for(let j=0;j<count;j++){
        const a=this.bands[ak[ak.length===1?0:j]],b=other.bands[bk[bk.length===1?0:j]],key=ak.length===1&&bk.length>1?bk[j]:ak[j];
        out[key]={v:a.v.map((v,i)=>+fn(v,b.v[i])),m:a.m.map((v,i)=>Math.min(v,b.m[i]))};
      }return new Img(out);
    }
    add(o){return this.binary(o,(a,b)=>a+b);}subtract(o){return this.binary(o,(a,b)=>a-b);}
    multiply(o){return this.binary(o,(a,b)=>a*b);}divide(o){return this.binary(o,(a,b)=>b===0?0:a/b);}
    pow(o){return this.binary(o,(a,b)=>a**b);}mod(o){return this.binary(o,(a,b)=>a%b);}
    min(o){return this.binary(o,Math.min);}max(o){return this.binary(o,Math.max);}
    abs(){return this.mapValues(Math.abs);}cos(){return this.mapValues(Math.cos);}sqrt(){return this.mapValues(Math.sqrt);}
    eq(o){return this.binary(o,(a,b)=>a===b);}neq(o){return this.binary(o,(a,b)=>a!==b);}
    lt(o){return this.binary(o,(a,b)=>a<b);}lte(o){return this.binary(o,(a,b)=>a<=b);}
    gt(o){return this.binary(o,(a,b)=>a>b);}gte(o){return this.binary(o,(a,b)=>a>=b);}
    and(o){return this.binary(o,(a,b)=>!!a&&!!b);}or(o){return this.binary(o,(a,b)=>!!a||!!b);}
    not(){return this.mapValues(a=>+!a);}
    // updateMask保留旧mask为0的位置；有效位置采用新mask值，不是min(旧值,新值)。
    updateMask(other){
      const masks=Object.values(other.bands),keys=Object.keys(this.bands);
      if(masks.length!==1&&masks.length!==keys.length)throw Error('Mask band arity mismatch');
      return new Img(Object.fromEntries(keys.map((k,j)=>{const b=this.bands[k],mask=masks[masks.length===1?0:j];
        return [k,{v:b.v.slice(),m:b.m.map((v,i)=>v>0&&mask.m[i]>0?mask.v[i]:0)}];
      })));
    }
    mask(){return new Img(Object.fromEntries(Object.entries(this.bands).map(([k,b])=>[k,{v:b.m.slice(),m:ones()}])));}
    unmask(value=0){return new Img(Object.fromEntries(Object.entries(this.bands).map(([k,b])=>[k,{v:b.v.map((v,i)=>b.m[i]?v:value),m:ones()}])));}
    where(test,value){
      if(!(value instanceof Img))value=Img.one(Array(n).fill(value));
      const t=test.first(),values=Object.values(value.bands);
      return new Img(Object.fromEntries(Object.entries(this.bands).map(([k,b],j)=>{
        const v=values[values.length===1?0:j];return [k,{v:b.v.map((x,i)=>t.m[i]&&t.v[i]&&v.m[i]?v.v[i]:x),m:b.m.slice()}];
      })));
    }
    translate(tx,ty){return new Img(Object.fromEntries(Object.entries(this.bands).map(([k,b])=>{
      const v=[],m=[];for(let i=0;i<n;i++){
        const x=i%width-tx,y=Math.floor(i/width)-ty,j=y*width+x,inside=x>=0&&x<width&&y>=0&&y<height;
        v.push(inside?b.v[j]:0);m.push(inside?b.m[j]:0);
      }return [k,{v,m}];
    })));}
    reduce(reducer){const bands=Object.values(this.bands),v=[],m=[];
      for(let i=0;i<n;i++){const values=bands.filter(b=>b.m[i]).map(b=>b.v[i]);m.push(+!!values.length);v.push(values.length?(reducer.kind==='max'?Math.max(...values):values.reduce((a,b)=>a+b,0)):0);}
      return Img.one(v,'reduced',m);
    }
  }
  const Image={constant(value){return Img.one(Array(n).fill(value));},
    pixelCoordinates(){return Img.one(Array.from({length:n},(_,i)=>coord(i)[0]+.5),'x').addBands(Img.one(Array.from({length:n},(_,i)=>coord(i)[1]+.5),'y'));},
    pixelLonLat(){return Img.one(Array.from({length:n},(_,i)=>30-(coord(i)[1]+.5)/3600),'latitude');},
    cat(images){return images.reduce((a,b)=>a.addBands(b));}};
  const ee={Image,Projection(crs,transform){return {crs,transform};},ImageCollection:{fromImages(images){return {sum(){
    const named=images.map((im,i)=>im.rename('band'+i));return Image.cat(named).reduce({kind:'sum'}).rename(Object.keys(images[0].bands)[0]);
  }};}},Reducer:{max(){return {kind:'max'};}}};
  return {ee,Img,n,coord};
};
