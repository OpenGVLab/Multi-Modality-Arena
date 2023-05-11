//点击切换 chat && voa
  var con = ".con1";
  //初始化一句话ai自由介绍
  setTimeout(function(){
    fasong(con +" .ai_list_l .ai_list ul","By using this service, users are required to agree to the following terms. ","ai","1");
  fasong(con +" .ai_list_r .ai_list ul"," lt must notbe used for anv illegal, harmful. violent racist. or sexual purposesPlease click the “Flag” button if you get any inappropriate answer! We will collectthose to keep improving our moderator.For an optimal experience, please use desktop computers for this demo, as mobiledevices may compromise its quality.","ai","1");
  },500)

  $(document).on('click','.nav li', function() {
 con = $(this).data("con");

 $(this).addClass("this").siblings("li").removeClass("this")
//  $(".con").hide()
//  $(con).show()
if(con == ".con1"){
    cropper_now = cropper;
   
}
if(con == ".con2"){
    cropper_now = cropper2
}
$(con).css({"display":"flex"}).siblings(".con").hide()
 
    })

 

//打字动画
 
function dazi(a) {
   
var d = a,
c = d.html(),  
 b = 0;
d.html("");
var e = setInterval(function () {
b++
d.html(c.substring(0, b) + (b & 1 ? '<i class="zwf">&nbsp;</i>' : '')); //&1 这个表达式 可以用来 判断 a的奇偶性
if (b >= c.length) {
    d.html(c.substring(0, b) + (b & 1 ? "" : "")); 
clearInterval(e);

}
 
},20)
return
}

//发送信息

function fasong(showdiv,itext,who,dh){
    // alert(itext)
    var iiximg = $(con + " .input img")
    var who = (!!who) ? who : "";
    if(itext || iiximg.length > 0){

if(iiximg.length > 0){
   var iiximg_src = iiximg.attr("src");
    var itext =   '<img src="'+iiximg_src+'">'  ;
    iiximg.remove();
    var who =  who + "img";
    cropper_now.clear()
};


        var t = '<li class="'+who+'"><div class="text">'+itext+'</div></li>'
        $(showdiv).append(t);

        if(dh){
        dazi($(showdiv).find("li:nth-last-of-type(1) .text"))
}
    


 if(who !== "ai"){
    //电脑回复
    var con_left = con +" .ai_list_l .ai_list ul";
var con_right = con +" .ai_list_r .ai_list ul"
    fasong(con_left,"no ","ai","1");
  fasong(con_right," ok.","ai","1");
 }
    

}
}
//按回车键发送用户消息
$('.con_r_1 .input input').on('keypress', function(e) {
        if (e.keyCode === 13) {
            var text = $(this).val();
            fasong(con + " .ai_list ul",text);
            $(this).val("")
        }
})
//清除用户输入的input消息
$(document).on('click','.i_clear', function() {
    $(this).siblings("input").removeClass("this").val("");
    $(this).siblings("img").remove();
    cropper_now.clear()
});

//按钮发送用户消息
$(document).on('click','.con_r_1>button', function() {
     var idiv =  $(this).siblings(".input").find("input")
    var itext = idiv.val()
    fasong(con + " .ai_list ul",itext);
    idiv.val("")
});

//检查用户输入文本显示清除按钮
$('.con_r_1 .input input').bind('input propertychange', function() {
($(this).val() !== "") ? $(this).addClass("this") : $(this).removeClass("this")
})

   
 //添加拖拽图片到图片框内
    function drop(e) {
        e.stopPropagation();
        e.preventDefault();
        e == e || window.event;//判断是浏览器图片还是电脑图片
        var files = e.dataTransfer.files;//获取拖拽的所有的文件
        for (var i = 0; i < files.length; i++) {
            var file = files[i];//获取每个文件
            if (file.type.indexOf("image") != -1) {//判断是否是图片
                var reader = new FileReader();//创建文件读取对象
                //读完回调事件
                reader.onload = function (f) {
                    // cropper.replace(this.result);
                tihuanimg(this.result)
                }
 
                reader.readAsDataURL(file);//把图片读成Base64编码字符串
            }
        }
    }

//图片上传清理默认事件
var ipt = $(".upfile");
    ipt.ondragover = function () {
        return false; //当元素移动至此  关闭默认处理
    }

 //选择图片到图片框内
 function previewImage(file){
  if (file.files &&file.files[0])
{
var reader =new FileReader();
reader.onload =function(evt){
  tihuanimg(evt.target.result);
  
}
reader.readAsDataURL(file.files[0]);
}


}
 
 
    //图片操作


    function getRoundedCanvas(sourceCanvas) {
      var canvas = document.createElement('canvas');
      var context = canvas.getContext('2d');
      var width = sourceCanvas.width;
      var height = sourceCanvas.height;

      canvas.width = width;
      canvas.height = height;
      context.imageSmoothingEnabled = true;
      context.drawImage(sourceCanvas, 0, 0, width, height);
      context.globalCompositeOperation = 'destination-in';
      context.beginPath();
      // context.arc(width / 2, height / 2, Math.min(width, height) / 2, 0, 2 * Math.PI, true);
      context.fill();
      return canvas;
    }

  


  var image = document.getElementById('image');
//初始化图片
var cropper = new Cropper(image, {
  viewMode: 1,
  dragMode: 'move',
  autoCropArea: 1,
  restore: false,
  modal: false,
  guides: false,
  highlight: false,
  cropBoxMovable: false,
  cropBoxResizable: false,
  toggleDragModeOnDblclick: false,
  croppable:false,
  ready: function () {
 
    cropper.croppable = false;
        cropper.clear();
        cropper.setDragMode("move")
  
  },
  cropend: function (e) {
    jianqie();
},
});


  var image2 = document.getElementById('image2');
//初始化图片
var cropper2 = new Cropper(image2, {
  viewMode: 1,
  dragMode: 'move',
  autoCropArea: 1,
  restore: false,
  modal: false,
  guides: false,
  highlight: false,
  cropBoxMovable: false,
  cropBoxResizable: false,
  toggleDragModeOnDblclick: false,
  croppable:false,
  ready: function () {
 
    cropper2.croppable = false;
    cropper2.clear();
    cropper2.setDragMode("move")
  
  },
  cropend: function (e) {
    jianqie();
},
});


//替换画布图片
function tihuanimg(img){
    cropper_now.replace(img);
    fasong(con +" .ai_list ul","<img src='"+img+"'>","img");
 $(con + " .con_l_0 .kin__1").hide()
 $(con + " .con_l_0 .kin__2").show()

}

//关闭图片
function gb_img(){
 $(con + " .con_l_0 .kin__1").show()
 $(con + " .con_l_0 .kin__2").hide()
 $(con + " .ylist dl.this").removeClass("this")
}

//剪切
function jianqie() {
  var croppedCanvas;
  var roundedCanvas;
  var roundedImage;

  if (!cropper_now.croppable) {
    return;
  }

  // Crop
  croppedCanvas = cropper_now.getCroppedCanvas();
  // Round
  roundedCanvas = getRoundedCanvas(croppedCanvas);

  // Show
  var ihtml = '<img src="'+roundedCanvas.toDataURL()+'">'
 
if($(con + " .input img").length > 0){
    $(con + " .input img").attr({"src":roundedCanvas.toDataURL()})
    
}else{
    $(con + " .input").prepend(ihtml) 
}
//   fasong(".ai_list ul",ihtml,"img")
};


//旋转图片
$(document).on("click",".rotate",function(){
 
    cropper_now.rotate(-90);

})

//移动图片
$(document).on("click",".move",function(){
 
    cropper_now.croppable = false;
cropper_now.setDragMode("move")
cropper_now.clear()

})
//选择examples图片到画布
$(document).on("click",".ylist dl",function(){
    var img = $(this).find("img").attr("src");
    $(this).addClass("this").siblings().removeClass("this")
    tihuanimg(img)
 
})
//切换剪切工具
$(document).on("click",".crop",function(){
    cropper_now.setDragMode("crop")
    cropper_now.croppable = true;
})

//清理剪切框
 $(document).on("click",".cropper-face button",function(){
    cropper_now.clear();
      $(con + " .con_r_1 .input img").remove()
    })

    $(document).on("click",".move",function(){
        $(this).addClass("this").siblings(".crop").removeClass("this")
    })

    $(document).on("click",".crop",function(){
        $(this).addClass("this").siblings(".move").removeClass("this")
    })
    //清空对话框
 $(document).on("click",".Clear",function(){

    })

    //点赞
  function myclass(a){
   return a.attr("class")
    }
    function text(c){

    }
    $(document).on("click",".con_r_2 button:not(.no)",function(){
       $(".itb").remove();
	  
        var that = $(this);
      var c = myclass(that)
      var con_left = con +" .ai_list_l";
var con_right = con +" .ai_list_r"
var con_center = con +" .con_r_0 .kin"
  
   if( c == "A_better") {
    $(con_left).append('<div class="itb tb_good myfirst">Better</div>');
   }
   if( c == "B_better") {
    $(con_right).append('<div class="itb tb_good myfirst">Better</div>');
    }
    if( c == "Tie") {
        $(con_center).append('<div class="itb tb_Tie myfirst2">Tie</div>');
    }
    if( c == "bad") {
        $(con_center).append('<div class="itb tb_bad myfirst">Both are bad</div>');
    }
	b = setInterval(off_open,1200);
    $(con +" .con_r_2 button:not(.Clear)").addClass("no")
     if( c == "Clear") {
        cropper_now.clear();
      $(con + " .ai_list ul *").remove();
      gb_img();
      $(con +" .con_r_2 button:not(.Clear)").removeClass("no")
      $(con +" .itb").remove()
    }
    })
	function off_open(){
		$(con +" .itb").addClass("this")
		clearInterval(b);
	}


//
//初始化画板
var cropper_now = cropper


 ////////////////////////////////////////////////////////////--------------------------------------


   //全屏功能
   var fullScreenClickCount=0;
    //调用各个浏览器提供的全屏方法
    var handleFullScreen = function () {
        var de = document.documentElement;

        if (de.requestFullscreen) {
            de.requestFullscreen();
        } else if (de.mozRequestFullScreen) {
            de.mozRequestFullScreen();
        } else if (de.webkitRequestFullScreen) {
            de.webkitRequestFullScreen();
        } else if (de.msRequestFullscreen) {
            de.msRequestFullscreen();
        }
        else {
            wtx.info("当前浏览器不支持全屏！");
        }

    };



    //调用各个浏览器提供的退出全屏方法
    var exitFullscreen=function() {
        if(document.exitFullscreen) {
            document.exitFullscreen();
        } else if(document.mozCancelFullScreen) {
            document.mozCancelFullScreen();
        } else if(document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        }
    }
    fullscreen = function ()  {
        if (fullScreenClickCount % 2 == 0) {
            handleFullScreen();
        } else {
            exitFullscreen();
        }
        fullScreenClickCount++;
    };


    (function($,h,c){var a=$([]),e=$.resize=$.extend($.resize,{}),i,k="setTimeout",j="resize",d=j+"-special-event",b="delay",f="throttleWindow";e[b]=250;e[f]=true;$.event.special[j]={setup:function(){if(!e[f]&&this[k]){return false}var l=$(this);a=a.add(l);$.data(this,d,{w:l.width(),h:l.height()});if(a.length===1){g()}},teardown:function(){if(!e[f]&&this[k]){return false}var l=$(this);a=a.not(l);l.removeData(d);if(!a.length){clearTimeout(i)}},add:function(l){if(!e[f]&&this[k]){return false}var n;function m(s,o,p){var q=$(this),r=$.data(this,d);r.w=o!==c?o:q.width();r.h=p!==c?p:q.height();n.apply(this,arguments)}if($.isFunction(l)){n=l;return m}else{n=l.handler;l.handler=m}}};function g(){i=h[k](function(){a.each(function(){var n=$(this),m=n.width(),l=n.height(),o=$.data(this,d);if(m!==o.w||l!==o.h){n.trigger(j,[o.w=m,o.h=l])}});g()},e[b])}})(jQuery,this);
 
    $(".ai_list ul").resize(function(){
        
        var h = $(this).height()
        var par = $(this).parent(".ai_list")
        par.animate({scrollTop:h}, 0);
 

  
    });