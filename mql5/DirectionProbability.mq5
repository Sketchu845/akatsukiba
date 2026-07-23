//+------------------------------------------------------------------+
//| DirectionProbability.mq5                                         |
//| Показывает над каждой свечой калиброванную вероятность успеха    |
//| входа buy/sell (модели LightGBM->ONNX, обучены на XAUUSDc M15).  |
//|                                                                  |
//| Файлы model_buy.onnx, model_sell.onnx, calibration.mqh должны    |
//| лежать в ОДНОЙ папке с этим .mq5 при компиляции.                 |
//| Вешать на график XAUUSDc M15.                                    |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_plots 0
#property strict

#resource "model_buy.onnx"  as uchar ExtBufBuy[]
#resource "model_sell.onnx" as uchar ExtBufSell[]
#include "calibration.mqh"

input int    InpBarsToShow = 300;   // На скольких последних свечах рисовать
input double InpEdgeCut    = 0.05;  // Порог сигнала: преимущество над базой (edge)
input bool   InpShowWeak   = false; // Показывать слабые метки (edge ниже порога)
input double InpTP_ATR     = 1.5;   // TP в ATR (как при обучении)
input double InpSL_ATR     = 1.0;   // SL в ATR (как при обучении)
input bool   InpDebugLog   = false; // Печатать вектор фичей последней свечи

#define ATR_P   14
#define RSI_P   14
#define EMA_P   50
#define WARMUP  450                 // прогрев рекурсивных индикаторов
#define H1NEED  1500                // сколько H1-баров тянем для контекста
#define PREFIX  "DPROB_"

long     g_hBuy  = INVALID_HANDLE;
long     g_hSell = INVALID_HANDLE;
datetime g_lastBar = 0;

//--- буферы расчёта (индекс 0 = самый старый бар)
MqlRates g_m15[];
double   g_atr[], g_rsi[], g_ema[], g_atrSma[];
MqlRates g_h1[];
double   g_h1atr[], g_h1ema[], g_h1rsi[];

//+------------------------------------------------------------------+
int OnInit()
  {
   g_hBuy  = OnnxCreateFromBuffer(ExtBufBuy,  ONNX_DEFAULT);
   g_hSell = OnnxCreateFromBuffer(ExtBufSell, ONNX_DEFAULT);
   if(g_hBuy == INVALID_HANDLE || g_hSell == INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer failed, err=", GetLastError());
      return INIT_FAILED;
     }

   const long inShape[]  = {1, FEATURE_COUNT};
   const long outLabel[] = {1};
   const long outProb[]  = {1, 2};
   if(!OnnxSetInputShape(g_hBuy, 0, inShape)  || !OnnxSetInputShape(g_hSell, 0, inShape) ||
      !OnnxSetOutputShape(g_hBuy, 0, outLabel)|| !OnnxSetOutputShape(g_hSell, 0, outLabel)||
      !OnnxSetOutputShape(g_hBuy, 1, outProb) || !OnnxSetOutputShape(g_hSell, 1, outProb))
     {
      Print("Onnx set shape failed, err=", GetLastError());
      return INIT_FAILED;
     }
   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(g_hBuy  != INVALID_HANDLE) OnnxRelease(g_hBuy);
   if(g_hSell != INVALID_HANDLE) OnnxRelease(g_hSell);
   ObjectsDeleteAll(0, PREFIX);
   Comment("");
  }
//+------------------------------------------------------------------+
//| Сглаживание Уайлдера = pandas ewm(alpha=1/p, adjust=False)       |
//+------------------------------------------------------------------+
void WilderFromTR(const double &src[], const int p, double &dst[])
  {
   int n = ArraySize(src);
   ArrayResize(dst, n);
   if(n == 0) return;
   dst[0] = src[0];
   double a = 1.0 / p;
   for(int i = 1; i < n; i++)
      dst[i] = dst[i-1] + a * (src[i] - dst[i-1]);
  }
//+------------------------------------------------------------------+
void CalcIndicators(const MqlRates &r[], double &atr[], double &rsi[], double &ema[])
  {
   int n = ArraySize(r);
   double tr[], gain[], loss[];
   ArrayResize(tr, n); ArrayResize(gain, n); ArrayResize(loss, n);

   tr[0] = r[0].high - r[0].low;
   gain[0] = 0; loss[0] = 0;
   for(int i = 1; i < n; i++)
     {
      double pc = r[i-1].close;
      tr[i] = MathMax(r[i].high - r[i].low,
              MathMax(MathAbs(r[i].high - pc), MathAbs(r[i].low - pc)));
      double d = r[i].close - pc;
      gain[i] = (d > 0 ?  d : 0);
      loss[i] = (d < 0 ? -d : 0);
     }
   WilderFromTR(tr, ATR_P, atr);

   double g[], l[];
   WilderFromTR(gain, RSI_P, g);
   WilderFromTR(loss, RSI_P, l);
   ArrayResize(rsi, n);
   for(int i = 0; i < n; i++)
      rsi[i] = (l[i] == 0.0 ? 100.0 : 100.0 - 100.0 / (1.0 + g[i] / l[i]));

   // EMA span=50 -> alpha = 2/(span+1), как pandas ewm(span=..., adjust=False)
   ArrayResize(ema, n);
   ema[0] = r[0].close;
   double a = 2.0 / (EMA_P + 1.0);
   for(int i = 1; i < n; i++)
      ema[i] = ema[i-1] + a * (r[i].close - ema[i-1]);
  }
//+------------------------------------------------------------------+
//| Собрать вектор фичей для M15-бара с индексом i (порядок = .mqh!) |
//| Возвращает false, если данных не хватает.                        |
//+------------------------------------------------------------------+
bool BuildFeatures(const int i, float &f[])
  {
   int n = ArraySize(g_m15);
   if(i < 108 || i >= n) return false;          // 100 для SMA(atr) + запас ret_8
   double atr = g_atr[i];
   if(atr <= 0) return false;

   // ret_1..ret_8
   int ks[5] = {1, 2, 3, 5, 8};
   for(int k = 0; k < 5; k++)
      f[k] = (float)((g_m15[i].close - g_m15[i - ks[k]].close) / atr);

   double o = g_m15[i].open, h = g_m15[i].high, l = g_m15[i].low, c = g_m15[i].close;
   double rng = h - l;
   f[5] = (float)((c - o) / atr);                              // body
   f[6] = (float)((h - MathMax(o, c)) / atr);                  // upper_wick
   f[7] = (float)((MathMin(o, c) - l) / atr);                  // lower_wick
   f[8] = (float)(rng > 0 ? (c - l) / rng : 0.5);              // close_pos
   f[9] = (float)((c - g_ema[i]) / atr);                       // dist_ema
   f[10] = (float)(g_rsi[i] / 100.0);                          // rsi
   f[11] = (float)(g_atrSma[i] > 0 ? atr / g_atrSma[i] : 1.0); // atr_rel

   double hi20 = g_m15[i].high, lo20 = g_m15[i].low;
   for(int j = i - 19; j <= i; j++)
     {
      hi20 = MathMax(hi20, g_m15[j].high);
      lo20 = MathMin(lo20, g_m15[j].low);
     }
   f[12] = (float)((hi20 - c) / atr);                          // dist_high20
   f[13] = (float)((c - lo20) / atr);                          // dist_low20

   MqlDateTime dt;
   TimeToStruct(g_m15[i].time, dt);
   double hour = dt.hour + dt.min / 60.0;
   f[14] = (float)MathSin(2.0 * M_PI * hour / 24.0);           // hour_sin
   f[15] = (float)MathCos(2.0 * M_PI * hour / 24.0);           // hour_cos
   int pyDow = (dt.day_of_week + 6) % 7;                       // Пн=0 как в pandas
   f[16] = (float)(pyDow / 4.0);                               // dow

   // --- H1: последний ЗАВЕРШЁННЫЙ час на момент закрытия M15-бара ---
   datetime m15close = g_m15[i].time + 900;
   int nh = ArraySize(g_h1);
   int j1 = -1;
   for(int j = nh - 1; j >= 0; j--)
      if(g_h1[j].time + 3600 <= m15close) { j1 = j; break; }
   if(j1 < 26) return false;                                   // нужно 24 бара диапазона + ret_3
   double hatr = g_h1atr[j1];
   if(hatr <= 0) return false;

   f[17] = (float)((g_h1[j1].close - g_h1ema[j1]) / hatr);     // h1_dist_ema
   f[18] = (float)((g_h1[j1].close - g_h1[j1 - 3].close) / hatr); // h1_ret_3
   f[19] = (float)(g_h1rsi[j1] / 100.0);                       // h1_rsi
   double hhi = g_h1[j1].high, hlo = g_h1[j1].low;
   for(int j = j1 - 23; j <= j1; j++)
     {
      hhi = MathMax(hhi, g_h1[j].high);
      hlo = MathMin(hlo, g_h1[j].low);
     }
   f[20] = (float)(hhi > hlo ? (g_h1[j1].close - hlo) / (hhi - hlo) : 0.5); // h1_range_pos
   return true;
  }
//+------------------------------------------------------------------+
bool Predict(const long handle, const float &f[], double &probOut)
  {
   matrixf x;
   x.Init(1, FEATURE_COUNT);
   for(int k = 0; k < FEATURE_COUNT; k++)
      x[0][k] = f[k];
   long   lbl[1];
   matrixf prob;
   prob.Init(1, 2);
   if(!OnnxRun(handle, ONNX_DEFAULT, x, lbl, prob))
     {
      Print("OnnxRun failed, err=", GetLastError());
      return false;
     }
   probOut = prob[0][1];
   return true;
  }
//+------------------------------------------------------------------+
void DrawLabel(const string name, const datetime t, const double price,
               const string text, const color clr, const int fontSize)
  {
   if(ObjectFind(0, name) < 0)
     {
      ObjectCreate(0, name, OBJ_TEXT, 0, t, price);
      ObjectSetString(0, name, OBJPROP_TEXT, text);
      ObjectSetInteger(0, name, OBJPROP_COLOR, clr);
      ObjectSetInteger(0, name, OBJPROP_FONTSIZE, fontSize);
      ObjectSetInteger(0, name, OBJPROP_ANCHOR, ANCHOR_LOWER);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
     }
  }
//+------------------------------------------------------------------+
void DrawSignalLines(const int i, const bool isBuy, const double atr)
  {
   double entry = g_m15[i].close;
   double tp = isBuy ? entry + InpTP_ATR * atr : entry - InpTP_ATR * atr;
   double sl = isBuy ? entry - InpSL_ATR * atr : entry + InpSL_ATR * atr;
   datetime t1 = g_m15[i].time, t2 = t1 + 900 * 16;

   string names[3] = {PREFIX+"line_tp", PREFIX+"line_sl", PREFIX+"line_en"};
   double lv[3] = {tp, sl, entry};
   color  cc[3] = {clrLimeGreen, clrOrangeRed, clrSilver};
   for(int k = 0; k < 3; k++)
     {
      ObjectDelete(0, names[k]);
      ObjectCreate(0, names[k], OBJ_TREND, 0, t1, lv[k], t2, lv[k]);
      ObjectSetInteger(0, names[k], OBJPROP_COLOR, cc[k]);
      ObjectSetInteger(0, names[k], OBJPROP_STYLE, k == 2 ? STYLE_DOT : STYLE_DASH);
      ObjectSetInteger(0, names[k], OBJPROP_SELECTABLE, false);
     }
  }
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[],
                const double &close[], const long &tick_volume[],
                const long &volume[], const int &spread[])
  {
   // работаем только по закрытию бара
   datetime cur = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(cur == g_lastBar) return rates_total;
   g_lastBar = cur;

   int need = InpBarsToShow + WARMUP;
   if(CopyRates(_Symbol, PERIOD_CURRENT, 1, need, g_m15) < need) return rates_total; // от бара 1 = только закрытые
   if(CopyRates(_Symbol, PERIOD_H1, 1, H1NEED, g_h1) < 100) return rates_total;

   CalcIndicators(g_m15, g_atr, g_rsi, g_ema);
   CalcIndicators(g_h1, g_h1atr, g_h1rsi, g_h1ema);   // (atr, rsi, ema)

   // SMA(100) от ATR
   int n = ArraySize(g_m15);
   ArrayResize(g_atrSma, n);
   double s = 0;
   for(int i = 0; i < n; i++)
     {
      s += g_atr[i];
      if(i >= 100) s -= g_atr[i - 100];
      g_atrSma[i] = (i >= 99 ? s / 100.0 : 0);
     }

   // рисуем последние InpBarsToShow закрытых свечей
   float f[FEATURE_COUNT];
   double bestP = 0; int bestI = -1; bool bestBuy = true;
   double lastPb = 0, lastPs = 0, lastEb = 0, lastEs = 0;
   bool haveLast = false;

   for(int i = MathMax(WARMUP, n - InpBarsToShow); i < n; i++)
     {
      string name = PREFIX + "t" + (string)(long)g_m15[i].time;
      if(ObjectFind(0, name) >= 0 && i < n - 1) continue;  // уже нарисовано
      if(!BuildFeatures(i, f)) continue;

      double pb = 0, ps = 0;
      if(!Predict(g_hBuy, f, pb) || !Predict(g_hSell, f, ps)) continue;
      pb = Calibrate(pb, CAL_BUY);
      ps = Calibrate(ps, CAL_SELL);

      double eb = pb - BASE_BUY;               // преимущество над своей базой
      double es = ps - BASE_SELL;
      bool   isBuy  = (eb >= es);
      double p      = isBuy ? pb : ps;
      double edge   = isBuy ? eb : es;
      bool   strong = (edge >= InpEdgeCut);
      color clr = isBuy ? (strong ? clrLime : clrDarkSeaGreen)
                        : (strong ? clrRed  : clrRosyBrown);
      string txt = (isBuy ? "B" : "S") + IntegerToString((int)MathRound(p * 100));
      if(strong || InpShowWeak)
         DrawLabel(name, g_m15[i].time, g_m15[i].high + 0.30 * g_atr[i], txt, clr,
                   strong ? 9 : 7);

      if(i == n - 1)
        { lastPb = pb; lastPs = ps; lastEb = eb; lastEs = es; haveLast = true; }

      if(i == n - 1 && strong)
        { bestP = p; bestI = i; bestBuy = isBuy; }

      if(i == n - 1 && InpDebugLog)
        {
         string dbg = "features[" + TimeToString(g_m15[i].time) + "]: ";
         for(int k = 0; k < FEATURE_COUNT; k++) dbg += DoubleToString(f[k], 6) + " ";
         Print(dbg);
         Print("raw->cal buy=", DoubleToString(pb, 4), " sell=", DoubleToString(ps, 4));
        }
     }

   if(bestI >= 0)
      DrawSignalLines(bestI, bestBuy, g_atr[bestI]);

   if(haveLast)
     {
      string side = (lastEb >= lastEs) ? "BUY" : "SELL";
      double edgeBest = MathMax(lastEb, lastEs);
      Comment(StringFormat(
         "DirectionProbability — вероятности с учётом стопов (TP %.1f / SL %.1f ATR, 4 часа)\n"
         "Последняя закрытая свеча:   BUY %.0f%%   SELL %.0f%%   (обычный уровень: %.0f%% / %.0f%%)\n"
         "Перевес над обычным: %s %+.1f п.п.   |   Сигнал (порог %.0f п.п.): %s",
         InpTP_ATR, InpSL_ATR,
         lastPb * 100, lastPs * 100, BASE_BUY * 100, BASE_SELL * 100,
         side, edgeBest * 100, InpEdgeCut * 100,
         edgeBest >= InpEdgeCut ? "ЕСТЬ" : "нет"));
     }

   ChartRedraw();
   return rates_total;
  }
//+------------------------------------------------------------------+
