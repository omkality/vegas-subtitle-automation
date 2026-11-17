using System;
using System.IO;
using System.Xml;
using System.Drawing;
using System.Text;
using System.Globalization;
using System.Collections.Generic;
using System.Reflection;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using ScriptPortal.Vegas;

public class EntryPoint {
  static string Root = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(ScriptPortal.Vegas.Script.File),"..","local","jobs","current"));
  static string Setting(string key) { XmlDocument d=new XmlDocument();d.Load(Path.Combine(Root,"edit-plan.xml"));return d.DocumentElement.GetAttribute(key); }
  static double D(string s) { return Double.Parse(s,CultureInfo.InvariantCulture); }
  static Timecode TC(double s) { return Timecode.FromSeconds(s); }
  static OFXParameter Param(OFXEffect fx,string name) { foreach(OFXParameter p in fx.Parameters) if(p.Name==name)return p; throw new Exception("Missing parameter: "+name); }
  static string Rtf(string text,bool blue) {
    StringBuilder b=new StringBuilder(@"{\rtf1\ansi\ansicpg1251\deff0{\fonttbl{\f0\fnil\fcharset204 Rubik;}}\viewkind4\uc1\pard\qc\lang1049\f0\fs"+(blue?"12 ":"16 "));
    foreach(char c in text) { if(c=='\n')b.Append(@"\par "); else if(c=='\\'||c=='{'||c=='}')b.Append('\\').Append(c); else if(c>127)b.Append(@"\u").Append((int)(short)c).Append('?'); else b.Append(c); }
    return b.Append(@"\par}").ToString();
  }
  static void SetStyle(OFXEffect fx,XmlElement generator) {
    foreach(XmlElement src in generator.SelectNodes("param")) {
      string name=src.GetAttribute("name"),type=src.GetAttribute("type"),v=src.GetAttribute("value"); OFXParameter p=Param(fx,name);
      if(name=="Text")continue;
      if(type=="OFXDoubleParameter")((OFXDoubleParameter)p).Value=D(v);
      else if(type=="OFXBooleanParameter")((OFXBooleanParameter)p).Value=Boolean.Parse(v);
      else if(type=="OFXStringParameter")((OFXStringParameter)p).Value=v;
      else if(type=="OFXRGBAParameter" || type=="OFXDouble2DParameter") {
        Dictionary<string,double> vals=new Dictionary<string,double>(); foreach(string pair in v.Split(',')) { if(pair.Length==0)continue; string[] kv=pair.Split('='); vals[kv[0]]=D(kv[1]); }
        if(type=="OFXRGBAParameter") { OFXColor c=new OFXColor();c.R=vals["R"];c.G=vals["G"];c.B=vals["B"];c.A=vals["A"];((OFXRGBAParameter)p).Value=c; }
        else { OFXDouble2D xy=new OFXDouble2D();xy.X=vals["X"];xy.Y=vals["Y"];((OFXDouble2DParameter)p).Value=xy; }
      }
    }
  }
  static Rectangle TextBounds(string path,string background) {
    using(Bitmap src=new Bitmap(path))using(Bitmap bg=new Bitmap(background))using(Bitmap bmp=new Bitmap(src.Width,src.Height,System.Drawing.Imaging.PixelFormat.Format32bppArgb))using(Bitmap clean=new Bitmap(src.Width,src.Height,System.Drawing.Imaging.PixelFormat.Format32bppArgb)) {
      using(Graphics g=Graphics.FromImage(bmp))g.DrawImageUnscaled(src,0,0);
      using(Graphics g=Graphics.FromImage(clean))g.DrawImageUnscaled(bg,0,0);
      BitmapData d=bmp.LockBits(new Rectangle(0,0,bmp.Width,bmp.Height),ImageLockMode.ReadOnly,System.Drawing.Imaging.PixelFormat.Format32bppArgb);
      byte[] data=new byte[d.Stride*bmp.Height];Marshal.Copy(d.Scan0,data,0,data.Length);bmp.UnlockBits(d);
      BitmapData bd=clean.LockBits(new Rectangle(0,0,clean.Width,clean.Height),ImageLockMode.ReadOnly,System.Drawing.Imaging.PixelFormat.Format32bppArgb);
      byte[] baseline=new byte[bd.Stride*clean.Height];Marshal.Copy(bd.Scan0,baseline,0,baseline.Length);clean.UnlockBits(bd);
      int x0=bmp.Width,x1=-1,y0=bmp.Height,y1=-1;
      for(int y=0;y<bmp.Height;y++)for(int x=0;x<bmp.Width;x++) { int i=y*d.Stride+x*4;if(Math.Abs(data[i]-baseline[i])+Math.Abs(data[i+1]-baseline[i+1])+Math.Abs(data[i+2]-baseline[i+2])>45) {x0=Math.Min(x0,x);x1=Math.Max(x1,x);y0=Math.Min(y0,y);y1=Math.Max(y1,y);} }
      if(x1<0)throw new Exception("Caption failed to render: "+path);
      return Rectangle.FromLTRB(x0,y0,x1+1,y1+1);
    }
  }
  static void CheckCaptionMargins(Vegas vegas) {
    string qa=Path.Combine(Root,"caption-qa");Directory.CreateDirectory(qa);
    try {
      int n=0;XmlWriterSettings settings=new XmlWriterSettings();settings.Indent=true;
      using(XmlWriter report=XmlWriter.Create(Path.Combine(Root,"caption-margins.xml"),settings)) {
        report.WriteStartElement("captions");report.WriteAttributeString("minimumMargin","0.12");
        for(int ti=0;ti<2;ti++)foreach(TrackEvent e in vegas.Project.Tracks[ti].Events) {
          string path=Path.Combine(qa,"caption-"+(n++).ToString("D3")+".png");
          OFXEffect fx=e.ActiveTake.Media.Generator.OFXEffect;
          OFXDouble2DParameter loc=(OFXDouble2DParameter)Param(fx,"Location");OFXDouble2D xy=loc.Value;xy.X=.5;loc.Value=xy;
          OFXDoubleParameter scale=(OFXDoubleParameter)Param(fx,"Scale");
          OFXRGBAParameter fill=(OFXRGBAParameter)Param(fx,"TextColor"),outline=(OFXRGBAParameter)Param(fx,"OutlineColor");OFXColor originalFill=fill.Value,originalOutline=outline.Value;
          Timecode at=e.Start+Timecode.FromMilliseconds(e.Length.ToMilliseconds()/2);
          Rectangle box=new Rectangle();
          for(int attempt=0;attempt<3;attempt++) {
            fx.AllParametersChanged();vegas.SaveSnapshot(path,ImageFileFormat.PNG,at);
            OFXColor transparentFill=originalFill,transparentOutline=originalOutline;transparentFill.A=0;transparentOutline.A=0;fill.Value=transparentFill;outline.Value=transparentOutline;fx.AllParametersChanged();
            string clean=Path.Combine(qa,"background.png");vegas.SaveSnapshot(clean,ImageFileFormat.PNG,at);
            fill.Value=originalFill;outline.Value=originalOutline;fx.AllParametersChanged();
            box=TextBounds(path,clean);
            if(box.Left>=vegas.Project.Video.Width*.12+2&&box.Right<=vegas.Project.Video.Width*.88-2)break;
            scale.Value*=vegas.Project.Video.Width*.72/(box.Width+28);
          }
          if(box.Left<vegas.Project.Video.Width*.12||box.Right>vegas.Project.Video.Width*.88)throw new Exception("Unsafe caption margin: "+e.Name);
          report.WriteStartElement("caption");report.WriteAttributeString("index",(n-1).ToString());report.WriteAttributeString("track",ti.ToString());report.WriteAttributeString("start",e.Start.ToMilliseconds().ToString(CultureInfo.InvariantCulture));report.WriteAttributeString("left",box.Left.ToString());report.WriteAttributeString("right",box.Right.ToString());report.WriteAttributeString("top",box.Top.ToString());report.WriteAttributeString("bottom",box.Bottom.ToString());report.WriteAttributeString("scale",scale.Value.ToString(CultureInfo.InvariantCulture));report.WriteString(e.Name);report.WriteEndElement();
        }
        report.WriteEndElement();
      }
    } finally {}
  }
  public void FromVegas(Vegas vegas) {
    File.WriteAllText(Path.Combine(Root,"build-error.txt"),"");
    File.WriteAllText(Path.Combine(Root,"build-result.txt"),"Started: "+DateTime.Now.ToString("O"));
    try {
      XmlDocument plan=new XmlDocument();plan.Load(Path.Combine(Root,"edit-plan.xml"));
      string output=plan.DocumentElement.GetAttribute("output");
      if(!String.Equals(Path.GetFullPath(vegas.Project.FilePath),Path.GetFullPath(Setting("source")),StringComparison.OrdinalIgnoreCase))throw new Exception("Unexpected project");
      if(vegas.Project.Video.Width!=1440||vegas.Project.Video.Height!=2560||Math.Abs(vegas.Project.Video.FrameRate-60000.0/1001)>.0001)throw new Exception("This profile requires 1440x2560 at 60000/1001 fps");
      double originalDuration=vegas.Project.Length.ToMilliseconds()/1000;
      if(vegas.Project.Tracks[0].Events.Count!=0 || vegas.Project.Tracks[1].Events.Count!=0)throw new Exception("Caption tracks must be empty");
      AudioTrack dialogue=(AudioTrack)vegas.Project.Tracks[3];
      AudioTrack beeps=(AudioTrack)vegas.Project.Tracks[4];
      if(beeps.Events.Count!=0)throw new Exception("The bleep destination track must be empty");
      if(plan.DocumentElement.GetAttribute("hotbar")=="true") {
        VideoTrack main=(VideoTrack)vegas.Project.Tracks[2];
        VideoTrack hotbar=new VideoTrack(vegas.Project,2,"Хотбар персонажа");vegas.Project.Tracks.Add(hotbar);
        using(Project template=vegas.ReadProject(Setting("referenceProject"))) {
          VideoEvent exemplar=(VideoEvent)template.Tracks[2].Events[0];
          foreach(TrackEvent clip in main.Events) {
            VideoEvent overlay=new VideoEvent(vegas.Project,clip.Start,clip.Length,"Хотбар");hotbar.Events.Add(overlay);
            Media media=Media.CreateInstance(vegas.Project,clip.ActiveTake.MediaPath);
            Take take=overlay.AddTake(media.GetVideoStreamByIndex(0));take.Offset=clip.ActiveTake.Offset;
            overlay.VideoMotion.ScaleToFill=exemplar.VideoMotion.ScaleToFill;
            overlay.MaintainAspectRatio=exemplar.MaintainAspectRatio;
            overlay.VideoMotion.Keyframes[0].Bounds=exemplar.VideoMotion.Keyframes[0].Bounds;
            foreach(Effect sourceFx in exemplar.Effects) {
              Effect targetFx=new Effect(vegas.VideoFX.FindChildByUniqueID(sourceFx.PlugIn.UniqueID));overlay.Effects.Add(targetFx);
              if(sourceFx.IsOFX)foreach(OFXParameter src in sourceFx.OFXEffect.Parameters) {
                PropertyInfo prop=src.GetType().GetProperty("Value");
                if(prop==null||!prop.CanRead||!prop.CanWrite)continue;
                OFXParameter dst=Param(targetFx.OFXEffect,src.Name);prop.SetValue(dst,prop.GetValue(src,null),null);
              }
              if(targetFx.IsOFX)targetFx.OFXEffect.AllParametersChanged();
            }
            overlay.PlaybackRate=clip.PlaybackRate;
            overlay.FadeIn.Length=clip.FadeIn.Length;overlay.FadeOut.Length=clip.FadeOut.Length;
            if(clip.IsGrouped)clip.Group.Add(overlay);
          }
        }
      }
      XmlDocument reference=new XmlDocument();reference.Load(Path.Combine(Root,"reference-style.xml"));
      vegas.Project.Tracks[0].Name="Товарищ · голубые субтитры";
      vegas.Project.Tracks[1].Name="Автор · белые субтитры";
      PlugInNode plugin=vegas.Generators.FindChildByUniqueID("{Svfx:com.vegascreativesoftware:titlesandtext}");
      if(plugin==null)throw new Exception("Titles & Text unavailable");
      foreach(XmlElement c in plan.SelectNodes("/edit/caption")) {
        bool blue=c.GetAttribute("speaker")=="friend"; int ti=blue?0:1;
        double start=D(c.GetAttribute("start")),end=D(c.GetAttribute("end"));string text=c.InnerText;
        if(start>=originalDuration||end>originalDuration+1001.0/60000)throw new Exception("Caption extends beyond the source timeline");
        end=Math.Min(end,originalDuration);
        Media m=Media.CreateInstance(vegas.Project,plugin); m.GetVideoStreamByIndex(0).Size=new Size(1080,1920);m.Length=TC(end-start);
        OFXEffect fx=m.Generator.OFXEffect;
        XmlElement gen=(XmlElement)reference.SelectSingleNode("/project/track[@index='"+ti+"']/event[1]/take/generator");
        SetStyle(fx,gen); ((OFXStringParameter)Param(fx,"Text")).Value=Rtf(text,blue);
        VideoEvent e=new VideoEvent(vegas.Project,TC(start),TC(end-start),text);vegas.Project.Tracks[ti].Events.Add(e);e.AddTake(m.GetVideoStreamByIndex(0));
        e.FadeIn.Length=TC(Math.Min(0.1001,(end-start)/5));e.FadeOut.Length=TC(Math.Min(0.1167833333,(end-start)/5));
      }
      CheckCaptionMargins(vegas);
      if(plan.SelectNodes("/edit/bleep").Count>0) {
        beeps.Name="Цензура · звуковой сигнал";
        Media beepMedia=Media.CreateInstance(vegas.Project,Setting("bleepSource"));
        foreach(XmlElement b in plan.SelectNodes("/edit/bleep")) {
          double start=D(b.GetAttribute("start")),end=D(b.GetAttribute("end"));
          List<TrackEvent> affected=new List<TrackEvent>();
          foreach(TrackEvent e in dialogue.Events) if(e.Start.ToMilliseconds()/1000<end && e.End.ToMilliseconds()/1000>start)affected.Add(e);
          foreach(TrackEvent e in affected) {
            TrackEvent muted=e;
            if(end<muted.End.ToMilliseconds()/1000-.0001) { TrackEvent right=muted.Split(TC(end)-muted.Start);right.FadeIn.Length=TC(.010); }
            if(start>muted.Start.ToMilliseconds()/1000+.0001) { TrackEvent right=muted.Split(TC(start)-muted.Start);muted.FadeOut.Length=TC(.010);muted=right; }
            muted.FadeIn.Gain=0;muted.FadeOut.Gain=0;
            muted.FadeIn.Length=TC(Math.Min(.010,(end-start)/4));muted.FadeOut.Length=TC(Math.Min(.010,(end-start)/4));
          }
          AudioEvent beep=new AudioEvent(vegas.Project,TC(start),TC(end-start),"Цензура");beeps.Events.Add(beep);
          Take bt=beep.AddTake(beepMedia.GetAudioStreamByIndex(0));bt.Offset=Timecode.FromMilliseconds(33.792);
          beep.FadeIn.Length=TC(.010);beep.FadeOut.Length=TC(.010);beep.FadeIn.Gain=.006578947f;beep.FadeOut.Gain=.006578947f;
        }
      }
      vegas.SaveProject(output);
      foreach(XmlElement snap in plan.SelectNodes("/edit/snapshot"))vegas.SaveSnapshot(snap.GetAttribute("path"),ImageFileFormat.PNG,TC(D(snap.GetAttribute("time"))));
      File.WriteAllText(Path.Combine(Root,"build-result.txt"),"Saved: "+output+"\nCaptions: "+plan.SelectNodes("/edit/caption").Count);
    } catch(Exception e) { File.WriteAllText(Path.Combine(Root,"build-error.txt"),e.ToString()); }
    vegas.NewProject(false,false);
    vegas.Exit();
  }
}
