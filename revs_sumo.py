# revs_sumo.py
import argparse, csv, math, os, random, xml.etree.ElementTree as ET
import traci
from veiculo import Veiculo

p=argparse.ArgumentParser()
p.add_argument("--lambda1",type=float,default=.5); p.add_argument("--lambda2",type=float,default=.3); p.add_argument("--lambda3",type=float,default=.2)
p.add_argument("--request-interval",type=int,default=20)
p.add_argument("--max-vehicles",type=int,default=400)
p.add_argument("--pricing-mode",choices=["static","dynamic"],default="dynamic")
p.add_argument("--pricing-alpha",type=float,default=.2); p.add_argument("--pricing-beta",type=float,default=.2); p.add_argument("--pricing-reference-interval",type=int,default=20)
p.add_argument("--revs-w1",type=float,default=.7); p.add_argument("--revs-w2",type=float,default=.3)
p.add_argument("--reputation-threshold",type=float,default=.5); p.add_argument("--communication-radius",type=float,default=250.)
p.add_argument("--direction-tolerance-deg",type=float,default=90.)
p.add_argument("--stay-time-mode",choices=["raw","normalized"],default="raw")
p.add_argument("--stay-time-normalization-cap",type=float,default=100.)
p.add_argument("--seed",type=int,default=42); p.add_argument("--experiment",default="c24_revs"); p.add_argument("--output-dir",default="."); p.add_argument("--headless",action="store_true")
a=p.parse_args()

L1,L2,L3=a.lambda1,a.lambda2,a.lambda3
REQ_INT=a.request_interval
MAXV=a.max_vehicles
PMODE,PALPHA,PBETA,PREF=a.pricing_mode,a.pricing_alpha,a.pricing_beta,a.pricing_reference_interval
RW1,RW2=a.revs_w1,a.revs_w2
RTH,RRADIUS,DIRTOL=a.reputation_threshold,a.communication_radius,a.direction_tolerance_deg
STMODE,STCAP=a.stay_time_mode,a.stay_time_normalization_cap
SEED,EXP,OUT=a.seed,a.experiment,a.output_dir
os.makedirs(OUT,exist_ok=True)

if abs(L1+L2+L3-1)>1e-6: raise ValueError("lambdas must sum to 1")
if abs(RW1+RW2-1)>1e-6: raise ValueError("REVS weights must sum to 1")
if REQ_INT<=0 or RRADIUS<=0 or STCAP<=0 or MAXV<=0: raise ValueError("invalid positive parameter")
if not 0<=RTH<=1: raise ValueError("reputation threshold must be in [0,1]")
if not 0<=DIRTOL<=180: raise ValueError("direction tolerance must be in [0,180]")

RNG_SCHEME="split_v2"
BENCHMARK_SCHEME="c24_revs_provider_selection_v1"
RREQ=random.Random(SEED+100000)
RVEH=random.Random(SEED+200000)
RREL=random.Random(SEED+300000)
RACT=random.Random(SEED+600000)
RENERGY=random.Random(SEED+900000)
RREQUESTER=random.Random(SEED+1000000)

CBMIN,CBMAX=3.,8.
RMAX=10.
SERV=[
 {"nome":"Processamento de Imagem Urbana","energia":18,"duracao":60,"preco":45.},
 {"nome":"Agregação de Dados de Sensores","energia":8,"duracao":30,"preco":22.},
 {"nome":"Reencaminhamento de Dados (Edge Relay)","energia":5,"duracao":25,"preco":15.},
 {"nome":"Atualização de Modelos Locais (ML)","energia":22,"duracao":80,"preco":60.},
 {"nome":"Diagnóstico Veicular","energia":10,"duracao":40,"preco":28.},
]
PMAX=max(s["preco"] for s in SERV)
ESMAX,DMAX,TMAX=25.,1000.,100.

vehicles={}
active_services=[]
total_requests=allocations=completed=successes=failures=0

REQF=os.path.join(OUT,f"requests_revs_{MAXV}_{EXP}.csv")
METF=os.path.join(OUT,f"metricas_revs_{MAXV}_{EXP}.csv")
OUTF=os.path.join(OUT,f"outcomes_revs_{MAXV}_{EXP}.csv")
SUMF=os.path.join(OUT,f"summary_revs_{MAXV}_{EXP}.csv")

COMMON=["experiment","method","seed","max_vehicles","rng_scheme","benchmark_scheme","pricing_mode","pricing_alpha","pricing_beta","pricing_reference_interval","request_interval","revs_w1","revs_w2","reputation_threshold","communication_radius","direction_tolerance_deg","stay_time_mode"]
def cv():
    return [EXP,"REVS-based",SEED,MAXV,RNG_SCHEME,BENCHMARK_SCHEME,PMODE,PALPHA,PBETA,PREF,REQ_INT,RW1,RW2,RTH,RRADIUS,DIRTOL,STMODE]

with open(REQF,"w",newline="",encoding="utf-8") as f:
    csv.writer(f).writerow(COMMON+["step","service","base_price","offered_price","legacy_task_pos_draw","active_vehicles","requester_vehicle","requester_x","requester_y","requester_speed","requester_angle","provider_pool","within_radius","same_direction","rep_threshold_pass","eligible_revs","allocated","failure_reason","selected_vehicle","selected_reputation_norm","selected_distance","selected_relative_speed","selected_stay_time","selected_revs_score","completion_step"])
with open(METF,"w",newline="",encoding="utf-8") as f:
    csv.writer(f).writerow(COMMON+["step","service","requester_vehicle","veiculo","reputacao_norm","energia_norm","energia","distancia","relative_speed","stay_time","stay_time_score_term","revs_score","candidate_count","base_price","offered_price"])
with open(OUTF,"w",newline="",encoding="utf-8") as f:
    csv.writer(f).writerow(COMMON+["selection_step","completion_step","vehicle","service","reliability_draw","success","reputation_before","reputation_after","reputation_delta"])

def clip(x,lo,hi): return max(lo,min(x,hi))
def sim_end(path):
    root=ET.parse(path).getroot(); node=root.find("./time/end")
    return int(float(node.attrib["value"])) if node is not None else 3600
def angdiff(a1,a2):
    d=abs(a1-a2)%360.; return min(d,360.-d)
def dist(p1,p2): return math.hypot(p1[0]-p2[0],p1[1]-p2[1])
def rnorm(v): return clip(float(v.reputacao)/RMAX,0.,1.)

def price(service,nactive):
    base=float(service["preco"])
    if PMODE=="static": return base,base
    demand=clip(PREF/float(REQ_INT)-1.,0.,1.)
    scarcity=1.-clip(nactive/float(MAXV),0.,1.)
    offered=clip(base*(1.+PALPHA*demand+PBETA*scarcity),base,PMAX)
    return base,offered

def stay(req,pro,remaining):
    d=dist(req.revs_pos,pro.revs_pos)
    if d>RRADIUS: return None
    rem=math.sqrt(max(RRADIUS**2-d**2,0.))
    rel=abs(req.revs_speed-pro.revs_speed)
    st=float(max(remaining,0)) if rel<=1e-12 else min(rem/rel,float(max(remaining,0)))
    return d,rel,st

def stayterm(st):
    return st if STMODE=="raw" else clip(st/STCAP,0.,1.)

def update_vehicles():
    ids=list(traci.vehicle.getIDList())
    if len(ids)>MAXV: ids=RACT.sample(ids,MAXV)
    out=[]
    for vid in ids:
        pos=traci.vehicle.getPosition(vid); sp=traci.vehicle.getSpeed(vid); ang=traci.vehicle.getAngle(vid)
        if vid not in vehicles:
            vehicles[vid]=Veiculo(
                vid,custo_base=RVEH.uniform(CBMIN,CBMAX),capacidade=RVEH.uniform(1,10),pos=pos[0],energia=RVEH.uniform(60,120),
                lambda1=L1,lambda2=L2,lambda3=L3,custo_max=CBMAX,preco_max=PMAX,energia_servico_max=ESMAX,distancia_max=DMAX,tempo_max=TMAX,
                initial_reputation=RVEH.uniform(5,10),rng_energy=RENERGY
            )
        v=vehicles[vid]
        v.posicao=pos[0]; v.revs_pos=(float(pos[0]),float(pos[1])); v.revs_speed=float(sp); v.revs_angle=float(ang)
        out.append(v)
    return out

def finish(step):
    global completed,successes,failures
    done=[s for s in active_services if s["fim"]<=step]
    for s in done:
        v=s["v"]; draw=RREL.random(); ok=draw>.1
        before=v.reputacao; v.atualizar_reputacao(ok); after=v.reputacao
        completed+=1; successes+=int(ok); failures+=int(not ok)
        with open(OUTF,"a",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(cv()+[s["inicio"],step,v.id,s["nome"],round(draw,9),ok,round(before,6),round(after,6),round(after-before,6)])
        active_services.remove(s)

def select_revs(req,active,service,step,endstep):
    pool=[v for v in active if v.id!=req.id and v.energia>=service["energia"]]
    radius=[]; direction=[]; repok=[]; eligible=[]
    for v in pool:
        sv=stay(req,v,endstep-step)
        if sv is None: continue
        radius.append(v)
        if angdiff(req.revs_angle,v.revs_angle)>DIRTOL: continue
        direction.append(v)
        rn=rnorm(v)
        if rn<RTH: continue
        repok.append(v)
        d,rel,st=sv; term=stayterm(st); score=RW1*rn+RW2*term
        eligible.append({"v":v,"rn":rn,"d":d,"rel":rel,"st":st,"term":term,"score":score})
    tel=[len(pool),len(radius),len(direction),len(repok),len(eligible)]
    if not eligible: return None,tel
    chosen=max(eligible,key=lambda c:(c["score"],c["rn"],c["st"],str(c["v"].id)))
    return chosen,tel

def request(step,active,endstep):
    global total_requests,allocations
    service=RREQ.choice(SERV)
    legacy_task_pos=RREQ.uniform(0,DMAX)  # preserves SHADAI request-stream consumption
    total_requests+=1
    base,offered=price(service,len(active))

    if len(active)<2:
        row=cv()+[step,service["nome"],base,offered,legacy_task_pos,len(active),"","","","","",0,0,0,0,0,False,"fewer_than_two_active_vehicles","","","","","","",""]
        with open(REQF,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow(row)
        return

    req=RREQUESTER.choice(active)
    chosen,tel=select_revs(req,active,service,step,endstep)
    if chosen is None:
        row=cv()+[step,service["nome"],base,offered,round(legacy_task_pos,6),len(active),req.id,req.revs_pos[0],req.revs_pos[1],req.revs_speed,req.revs_angle,*tel,False,"no_revs_candidate","","","","","","",""]
        with open(REQF,"a",newline="",encoding="utf-8") as f: csv.writer(f).writerow(row)
        return

    v=chosen["v"]; allocations+=1; fin=step+service["duracao"]
    with open(METF,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(cv()+[step,service["nome"],req.id,v.id,round(chosen["rn"],6),round(clip(v.energia/120.,0.,1.),6),round(v.energia,6),round(chosen["d"],6),round(chosen["rel"],6),round(chosen["st"],6),round(chosen["term"],6),round(chosen["score"],6),tel[-1],round(base,6),round(offered,6)])
    with open(REQF,"a",newline="",encoding="utf-8") as f:
        csv.writer(f).writerow(cv()+[step,service["nome"],round(base,6),round(offered,6),round(legacy_task_pos,6),len(active),req.id,req.revs_pos[0],req.revs_pos[1],req.revs_speed,req.revs_angle,*tel,True,"",v.id,round(chosen["rn"],6),round(chosen["d"],6),round(chosen["rel"],6),round(chosen["st"],6),round(chosen["score"],6),fin])
    active_services.append({"v":v,"inicio":step,"fim":fin,"nome":service["nome"]})
    v.executar_servico(service)

def summary():
    ar=allocations/total_requests if total_requests else 0.
    sr=successes/completed if completed else 0.
    with open(SUMF,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(COMMON+["total_requests","successful_allocations","allocation_rate","completed_tasks","successful_tasks","failed_tasks","task_success_rate","pending_tasks"])
        w.writerow(cv()+[total_requests,allocations,ar,completed,successes,failures,sr,len(active_services)])
    return ar,sr

def run(cfg):
    endstep=sim_end(cfg); binary="sumo" if a.headless else "sumo-gui"
    traci.start([binary,"-c",cfg,"--seed",str(SEED)])
    for step in range(endstep):
        traci.simulationStep(); active=update_vehicles(); finish(step)
        if step>0 and step%REQ_INT==0: request(step,active,endstep)
    traci.close()
    ar,sr=summary()
    print(f"REVS-based | seed={SEED} | allocation={ar*100:.2f}% | success={sr*100:.2f}%")

if __name__=="__main__":
    run("cidade.sumocfg")
