function sortTable(idx){
    var otable=document.querySelector('table'), 
        obody=otable.tBodies[0], 
        otr=obody.rows,
        tarr=[];
    for(var i=1;i<otr.length;i++){
        tarr[i-1]=otr[i];
    }
    if(obody.sortCol==idx){
        tarr.reverse();
    }else{
        tarr.sort(function(tr1,tr2){
            var value1=tr1.cells[idx].innerHTML;
            var value2=tr2.cells[idx].innerHTML;
            if(!isNaN(value1)&&!isNaN(value2)){
                return value1-value2;
            }else{
                return value1.localeCompare(value2);
            }
        })
    }
    var fragment=document.createDocumentFragment();
    for(var i=0;i<tarr.length;i++){
        fragment.appendChild(tarr[i]);
    }
    obody.appendChild(fragment);
    obody.sortCol=idx;
}

function DragSort(){
    var otbox=document.querySelector('.table-box'), 
        otable=document.querySelector('table'), 
        obody=otable.tBodies[0], 
        oth=obody.getElementsByTagName('th'), 
        otd=obody.getElementsByTagName('td'), 
        box=document.querySelector('.box'),
        arrn=[]; 
    for(var i=0;i<otd.length;i++){
        otd[i].onmousedown=function(e){
            var e=e||window.event, 
                target=e.target,
                thW=target.offsetWidth, 
                maxL=otbox.offsetWidth-thW, 
                rows=otable.rows, 
                tboxL=otbox.offsetLeft, 
                disX=target.offsetLeft, 
                that=this, 
                cdisX=e.clientX-tboxL-disX; 
            for(var i=0;i<rows.length;i++){
                var op=document.createElement('p');
                op.innerHTML=rows[i].cells[this.cellIndex].innerHTML;
                box.appendChild(op);
            }
            for(var i=0;i<oth.length;i++){
                arrn.push(oth[i].offsetLeft);
            }
            box.style.display='block';
            box.style.width=thW+'px';
            box.style.left=disX+'px';

            document.onmousemove=function(e){
                box.style.top=0;
                box.style.left=e.clientX-tboxL-cdisX+'px';
                if(box.offsetLeft>maxL){
                    box.style.left=maxL+'px';
                }else if(box.offsetLeft<0){
                    box.style.left=0;
                }
                document.onselectstart=function(){return false;}
            }
            document.onmouseup=function(e){
                var ops=box.getElementsByTagName('p'),
                    oboxL=box.offsetLeft+cdisX;
                for(var i=0;i<arrn.length;i++){
                    if(arrn[i]<oboxL){
                        var index=i;
                    }
                }
                for(var i=0;i<rows.length;i++){
                    rows[i].cells[that.cellIndex].innerHTML='';
                    rows[i].cells[that.cellIndex].innerHTML=rows[i].cells[index].innerHTML;
                    rows[i].cells[index].innerHTML='';
                    rows[i].cells[index].innerHTML=ops[i].innerHTML;
                }
                box.innerHTML='';
                arrn=[];
                box.style.display='none';
                document.onmousemove=null;
                document.onmouseup=null;
                document.onselectstart=function(){return false;}
            }
        }
    }
}

DragSort();