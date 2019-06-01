import torch as pt
import contrib

device = contrib.get_device()

X_train = pt.tensor([
    [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.3803921937942505], [0.3764706254005432], [0.3019607961177826], [0.46274513006210327], [0.2392157018184662], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.3529411852359772], [0.5411764979362488], [0.9215686917304993], [0.9215686917304993], [0.9215686917304993], [0.9215686917304993], [0.9215686917304993], [0.9215686917304993], [0.9843137860298157], [0.9843137860298157], [0.9725490808486938], [0.9960784912109375], [0.960784375667572], [0.9215686917304993], [0.7450980544090271], [0.08235294371843338], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.5490196347236633], [0.9843137860298157], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.7411764860153198], [0.09019608050584793], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.8862745761871338], [0.9960784912109375], [0.8156863451004028], [0.7803922295570374], [0.7803922295570374], [0.7803922295570374], [0.7803922295570374], [0.545098066329956], [0.2392157018184662], [0.2392157018184662], [0.2392157018184662], [0.2392157018184662], [0.2392157018184662], [0.501960813999176], [0.8705883026123047], [0.9960784912109375], [0.9960784912109375], [0.7411764860153198], [0.08235294371843338], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.14901961386203766], [0.32156863808631897], [0.05098039656877518], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.13333334028720856], [0.8352941870689392], [0.9960784912109375], [0.9960784912109375], [0.45098042488098145], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.3294117748737335], [0.9960784912109375], [0.9960784912109375], [0.917647123336792], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.3294117748737335], [0.9960784912109375], [0.9960784912109375], [0.917647123336792], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.41568630933761597], [0.615686297416687], [0.9960784912109375], [0.9960784912109375], [0.9529412388801575], [0.20000001788139343], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.09803922474384308], [0.458823561668396], [0.8941177129745483], [0.8941177129745483], [0.8941177129745483], [0.9921569228172302], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9411765336990356], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.2666666805744171], [0.46666669845581055], [0.8627451658248901], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.5568627715110779], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.14509804546833038], [0.7333333492279053], [0.9921569228172302], [0.9960784912109375], [0.9960784912109375], [0.9960784912109375], [0.874509871006012], [0.8078432083129883], [0.8078432083129883], [0.29411765933036804], [0.2666666805744171], [0.8431373238563538], [0.9960784912109375], [0.9960784912109375], [0.458823561668396], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.4431372880935669], [0.8588235974311829], [0.9960784912109375], [0.9490196704864502], [0.8901961445808411], [0.45098042488098145], [0.3490196168422699], [0.12156863510608673], [0.0], [0.0], [0.0], [0.0], [0.7843137979507446], [0.9960784912109375], [0.9450981020927429], [0.16078431904315948], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.6627451181411743], [0.9960784912109375], [0.6901960968971252], [0.24313727021217346], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.1882353127002716], [0.9058824181556702], [0.9960784912109375], [0.917647123336792], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.07058823853731155], [0.4862745404243469], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.3294117748737335], [0.9960784912109375], [0.9960784912109375], [0.6509804129600525], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.545098066329956], [0.9960784912109375], [0.9333333969116211], [0.22352942824363708], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.8235294818878174], [0.9803922176361084], [0.9960784912109375], [0.658823549747467], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]],
    
    [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.9490196704864502], [0.9960784912109375], [0.9372549653053284], [0.22352942824363708], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.3490196168422699], [0.9843137860298157], [0.9450981020927429], [0.33725491166114807], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.019607843831181526], [0.8078432083129883], [0.9647059440612793], [0.615686297416687], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.01568627543747425], [0.458823561668396], [0.2705882489681244], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.12156863510608673], [0.5176470875740051], [0.9960784912109375], [0.9921569228172302], [0.9960784912109375], [0.8352941870689392], [0.32156863808631897], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.08235294371843338], [0.5568627715110779], [0.9137255549430847], [0.988235354423523], [0.9921569228172302], [0.988235354423523], [0.9921569228172302], [0.988235354423523], [0.874509871006012], [0.0784313753247261], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.48235297203063965], [0.9960784912109375], [0.9921569228172302], [0.9960784912109375], [0.9921569228172302], [0.8784314393997192], [0.7960785031318665], [0.7960785031318665], [0.874509871006012], [1.0], [0.8352941870689392], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.7960785031318665], [0.9921569228172302], [0.988235354423523], [0.9921569228172302], [0.8313726186752319], [0.0784313753247261], [0.0], [0.0], [0.2392157018184662], [0.9921569228172302], [0.988235354423523], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.16078431904315948], [0.9529412388801575], [0.8784314393997192], [0.7960785031318665], [0.7176470756530762], [0.16078431904315948], [0.5960784554481506], [0.11764706671237946], [0.0], [0.0], [1.0], [0.9921569228172302], [0.40000003576278687], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.1568627506494522], [0.0784313753247261], [0.0], [0.0], [0.40000003576278687], [0.9921569228172302], [0.19607844948768616], [0.0], [0.32156863808631897], [0.9921569228172302], [0.988235354423523], [0.0784313753247261], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.32156863808631897], [0.8392157554626465], [0.12156863510608673], [0.4431372880935669], [0.9137255549430847], [0.9960784912109375], [0.9137255549430847], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.24313727021217346], [0.40000003576278687], [0.32156863808631897], [0.16078431904315948], [0.9921569228172302], [0.9098039865493774], [0.9921569228172302], [0.988235354423523], [0.9137255549430847], [0.19607844948768616], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.5960784554481506], [0.9921569228172302], [0.9960784912109375], [0.9921569228172302], [0.9960784912109375], [0.9921569228172302], [0.9960784912109375], [0.9137255549430847], [0.48235297203063965], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.5960784554481506], [0.988235354423523], [0.9921569228172302], [0.988235354423523], [0.9921569228172302], [0.988235354423523], [0.7529412508010864], [0.19607844948768616], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.24313727021217346], [0.7176470756530762], [0.7960785031318665], [0.9529412388801575], [0.9960784912109375], [0.9921569228172302], [0.24313727021217346], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.1568627506494522], [0.6745098233222961], [0.988235354423523], [0.7960785031318665], [0.0784313753247261], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.08235294371843338], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.7176470756530762], [0.9960784912109375], [0.4392157196998596], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.24313727021217346], [0.7960785031318665], [0.6392157077789307], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.2392157018184662], [0.9921569228172302], [0.5921568870544434], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.08235294371843338], [0.8392157554626465], [0.7529412508010864], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.04313725605607033], [0.8352941870689392], [0.9960784912109375], [0.5921568870544434], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.40000003576278687], [0.9921569228172302], [0.5921568870544434], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.16078431904315948], [0.8352941870689392], [0.988235354423523], [0.9921569228172302], [0.43529415130615234], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.16078431904315948], [1.0], [0.8352941870689392], [0.3607843220233917], [0.20000001788139343], [0.0], [0.0], [0.12156863510608673], [0.3607843220233917], [0.6784313917160034], [0.9921569228172302], [0.9960784912109375], [0.9921569228172302], [0.5568627715110779], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.6745098233222961], [0.988235354423523], [0.9921569228172302], [0.988235354423523], [0.7960785031318665], [0.7960785031318665], [0.9137255549430847], [0.988235354423523], [0.9921569228172302], [0.988235354423523], [0.9921569228172302], [0.5098039507865906], [0.0784313753247261], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.08235294371843338], [0.7960785031318665], [1.0], [0.9921569228172302], [0.9960784912109375], [0.9921569228172302], [0.9960784912109375], [0.9921569228172302], [0.9568628072738647], [0.7960785031318665], [0.32156863808631897], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0784313753247261], [0.5921568870544434], [0.5921568870544434], [0.9921569228172302], [0.6705882549285889], [0.5921568870544434], [0.5921568870544434], [0.1568627506494522], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]]
], dtype=pt.double, device=device)

Y_train = pt.tensor([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
], dtype=pt.double, device=device)