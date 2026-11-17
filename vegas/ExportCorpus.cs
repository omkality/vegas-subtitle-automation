using System;
using System.IO;
using System.Xml;
using System.Text;
using System.Globalization;
using System.Windows.Forms;
using System.Collections.Generic;
using ScriptPortal.Vegas;

public static class CorpusCatalog {
  static string Repo = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(ScriptPortal.Vegas.Script.File), ".."));
  static XmlElement Settings { get { XmlDocument d=new XmlDocument();d.Load(Path.Combine(Repo,"local","settings.xml"));return d.DocumentElement; } }
  static string Root { get { return Settings.GetAttribute("corpusDirectory"); } }
  static string F(double n){return n.ToString("R",CultureInfo.InvariantCulture);}
  static double S(Timecode t){return t.ToMilliseconds()/1000;}
  static void A(XmlWriter w,string key,object value){w.WriteAttributeString(key,value==null?"":value.ToString());}
  static OFXParameter Find(OFXEffect fx,string name){foreach(OFXParameter p in fx.Parameters)if(p.Name==name)return p;return null;}
  public static void Export(Vegas vegas) {
    Directory.CreateDirectory(Root);Directory.CreateDirectory(Path.Combine(Root,"projects"));
    foreach(string file in Directory.GetFiles(Settings.GetAttribute("projectsDirectory"),"*.veg")) {
      string name=Path.GetFileNameWithoutExtension(file);
      bool skip=false;foreach(XmlElement x in Settings.SelectNodes("exclude"))if(name.StartsWith(x.InnerText,StringComparison.OrdinalIgnoreCase))skip=true;
      if(skip)continue;
      string output=Path.Combine(Root,"projects",name+".xml");
      if(File.Exists(output)&&new FileInfo(output).Length>100)continue;
      File.AppendAllText(Path.Combine(Root,"progress.log"),DateTime.Now.ToString("O")+" START "+name+"\n");
      try {
        using(Project p=vegas.ReadProject(file)) {
          XmlWriterSettings settings=new XmlWriterSettings();settings.Indent=true;
          using(XmlWriter w=XmlWriter.Create(output,settings))using(RichTextBox rtf=new RichTextBox()) {
            w.WriteStartElement("project");A(w,"path",file);A(w,"duration",F(S(p.Length)));A(w,"fps",F(p.Video.FrameRate));A(w,"width",p.Video.Width);A(w,"height",p.Video.Height);
            int captions=0;
            foreach(Track track in p.Tracks) {
              w.WriteStartElement("track");A(w,"index",track.Index);A(w,"type",track.MediaType);A(w,"name",track.Name);A(w,"mute",track.Mute);
              if(track.IsAudio())A(w,"volume",F(((AudioTrack)track).Volume));
              foreach(TrackEvent e in track.Events) {
                w.WriteStartElement("event");A(w,"index",e.Index);A(w,"start",F(S(e.Start)));A(w,"end",F(S(e.End)));A(w,"rate",F(e.PlaybackRate));A(w,"mute",e.Mute);A(w,"fadeIn",F(S(e.FadeIn.Length)));A(w,"fadeOut",F(S(e.FadeOut.Length)));A(w,"gain",F(e.FadeIn.Gain));
                if(e.ActiveTake!=null) {
                  Take take=e.ActiveTake;Media m=take.Media;A(w,"path",take.MediaPath);A(w,"offset",F(S(take.Offset)));A(w,"stream",take.StreamIndex);
                  if(m.Generator!=null) {
                    Effect fx=m.Generator;w.WriteStartElement("generator");A(w,"plugin",fx.PlugIn.Name);A(w,"id",fx.PlugIn.UniqueID);
                    if(fx.IsOFX) {
                      OFXStringParameter text=Find(fx.OFXEffect,"Text") as OFXStringParameter;
                      if(text!=null) {
                        string raw=text.Value,plain;
                        try{rtf.Rtf=raw;plain=rtf.Text;}catch{plain=raw;}
                        OFXRGBAParameter color=Find(fx.OFXEffect,"TextColor") as OFXRGBAParameter;
                        string speaker="unknown";
                        if(color!=null){OFXColor c=color.Value;A(w,"r",F(c.R));A(w,"g",F(c.G));A(w,"b",F(c.B));A(w,"a",F(c.A));if(c.G>.85&&c.B>.85&&c.R<.9)speaker="friend";else if(c.R>.9&&c.G>.9&&c.B>.9)speaker="self";}
                        if(plain.Trim().StartsWith("*")&&plain.Trim().EndsWith("*"))speaker="friend";
                        A(w,"speaker",speaker);w.WriteElementString("text",plain);w.WriteElementString("rtf",raw);captions++;
                      }
                    }
                    w.WriteEndElement();
                  }
                  else A(w,"exists",File.Exists(take.MediaPath));
                }
                w.WriteEndElement();
              }
              w.WriteEndElement();
            }
            w.WriteEndElement();
            File.AppendAllText(Path.Combine(Root,"progress.log"),DateTime.Now.ToString("O")+" OK "+name+" captions="+captions+"\n");
          }
        }
        GC.Collect();GC.WaitForPendingFinalizers();
      } catch(Exception ex) {File.AppendAllText(Path.Combine(Root,"errors.log"),name+"\n"+ex+"\n");}
    }
    File.WriteAllText(Path.Combine(Root,"complete.txt"),DateTime.Now.ToString("O"));
  }
}
public class EntryPoint {
  public void FromVegas(Vegas vegas){CorpusCatalog.Export(vegas);vegas.NewProject(false,false);vegas.Exit();}
}
