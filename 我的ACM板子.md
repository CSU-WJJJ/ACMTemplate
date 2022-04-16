# 我的ACM板子

## 基础算法

### 快读

```c++
inline char getch() {
    static char buf[1 << 14], *p1 = buf, *p2 = buf;
    return p1 == p2 && (p2 = (p1 = buf) + fread(buf, 1, 1 << 14, stdin), p1 == p2) ? EOF : *p1++;
}

template <typename T>
void read(T &x) {
    static char c;
    static bool iosig;
    for(c = getch(), iosig = false;!isdigit(c);c = getch()) {
        if(c == -1) { return; }
        iosig |= c == '-';
    }
    for(x = 0; isdigit(c); c = getch()) x = x * 10 + (c ^ '0');
    if(iosig) x = -x;
}
```

### 二分

```cpp
// 满足ok(x)的最小x
int l=1,r=n;
while(l<r){
    int mid=l+r>>1;
    if(ok(mid)){
        r=mid;
    }else{
        l=mid+1;
    }
}
// 满足ok(x)的最大x
int l=1,r=n;
while(l<r){
    int mid=l+r+1>>1;
    if(ok(mid)){
        l=mid;
    }else{
        r=mid-1;
    }
}
//最后l==r即为答案 
```

###  离散化

```cpp
int a[10050],b[10050];
vector<int>alls;
int findx(int x)
{
    return lower_bound(alls.begin(),alls.end(),x)-alls.begin()+1;
}
for(int i=1;i<=m;i++){
    cin>>a[i]>>b[i];
    alls.push_back(a[i]); alls.push_back(b[i]);
}
sort(alls.begin(),alls.end());
alls.erase(unique(alls.begin(),alls.end()),alls.end());
```

### ST表

```cpp
int n,m;
int a[100050];
int st[100050][100];
//st[i][j]为从位置i开始连续(1<<j)个数的最值 
void init()
{
	for(int i=1;i<=n;i++){
		st[i][0]=a[i];
	}
	for(int j=1;(1<<j)<=n;j++){
		for(int i=1;i+(1<<j)-1<=n;i++){
			st[i][j]=max(st[i][j-1],st[i+(1<<(j-1))][j-1]);
		}
	}
}
int search(int l,int r)
{
	int k=log2(r-l+1);
	return max(st[l][k],st[r-(1<<k)+1][k]);	
}
```

### 倍增

#### 求LCA

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define inf 0x3f3f3f3f
#define endl '\n'
int n,m,s,head[500050];
int fa[500050];
struct edge{
    int nxt,to;
}e[1000050];
int cnt,dep[500050],p[500050][21];
void add(int u,int v)
{
    e[++cnt].to=v;
    e[cnt].nxt=head[u];
    head[u]=cnt;
}
void dfs(int u,int fa)
{
    dep[u]=dep[fa]+1;
    p[u][0]=fa;
    for(int i=1;(1<<i)<=dep[u];i++){
        p[u][i]=p[p[u][i-1]][i-1];
    }
    for(int i=head[u];i;i=e[i].nxt){
        int v=e[i].to;
        if(v!=fa){
            dfs(v,u);
        }
    }
}
int lca(int a,int b)
{
    if(dep[a]>dep[b]){
        swap(a,b);
    }
    for(int i=20;i>=0;i--){
        if(dep[a]<=dep[b]-(1<<i)){
            b=p[b][i];
        }
    }
    if(a==b) return a;
    for(int i=20;i>=0;i--){
        if(p[a][i]==p[b][i]){
            continue;
        }
        else{
            a=p[a][i],b=p[b][i];
        }
    }
    return p[a][0];
}
int main(){
ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
cin>>n>>m>>s; // s为树根的序号
for(int i=1;i<n;i++){
    int x,y;
    cin>>x>>y;
    add(x,y); add(y,x);
}
dfs(s,0);
for(int i=1;i<=m;i++){
    int x,y;
    cin>>x>>y;
    cout<<lca(x,y)<<endl;
}

    return 0;
}
```

## 字符串

### KMP

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define maxn 1000050
int nextt[maxn];
int n,m;
char a[maxn],b[maxn];
int aa,bb;
void GetNext()
{
    int j=0; nextt[1]=0;
    for(int i=2;i<=bb;i++){
        while(j&&b[i]!=b[j+1]) j=nextt[j];
        if(b[j+1]==b[i]) j++;
        nextt[i]=j;
    }
}
void kmp()
{
    int j=0;
    for(int i=1;i<=aa;i++){
        while(j&&b[j+1]!=a[i]) j=nextt[j];
        if(b[j+1]==a[i]) j++;
        if(j==bb){
            //匹配成功
            cout<<i-bb+1<<endl; j=nextt[j];
        }
    }
}
int main(){
cin>>a+1>>b+1;
aa=strlen(a+1); bb=strlen(b+1);
GetNext(); kmp();
for(int i=1;i<=bb;i++){
    cout<<nextt[i]<<" ";
}
    return 0;
}
```

### 扩展KMP

![QQ图片20210823163357](C:\Users\86185\Desktop\QQ图片20210823163357.png)

```c++
#include<bits/stdc++.h>
const int maxn = 2e7 + 1;
const int& max2(const int& a,const int& b){return a > b ? a : b;}
char s[maxn],t[maxn];
int n,m,nxt[maxn],ext[maxn];
void calc_nxt(){
	nxt[0] = n;
	int j = 0;
	while(j + 1 < n && t[j] == t[j + 1]) ++j;
	nxt[1] = j; // nxt[0],nxt[1] 暴力算
	int k = 1; // 我是把k放在循环外，p用k得到，更新的也是k
	for(int i = 2;i < n;++i){
		int p = k + nxt[k] - 1;
		if(i + nxt[i - k] <= p) nxt[i] = nxt[i - k]; // 情况一
		else {
			j = max2(p - i + 1,0); // 情况二
			while(i + j < n && t[i + j] == t[j]) ++j; // 情况二，暴力一位一位比对
			nxt[i] = j,k = i; // 记得更新k
		}
	}
}
void calc_ext(){// 和刚才几乎一样
	int j = 0;
	while(j < n && j < m && s[j] == t[j]) ++j;
	ext[0] = j; // ext[0] 暴力算
	int k = 0; // 放在循环外　
	for(int i = 1;i < m;++i){
		int p = k + ext[k] - 1;
		if(i + nxt[i - k] <= p) ext[i] = nxt[i - k];// 省略讲解里的y
		else {
			j = max2(p - i + 1,0); // 取max
			while(i + j < m && j < n && s[i + j] == t[j]) ++j; // 一位一位比对
			ext[i] = j,k = i;
		}
	}
} 
int main(){
	scanf("%s%s",s,t);
	n = strlen(t),m = strlen(s);
	calc_nxt();calc_ext();
	long long res1 = 0,res2 = 0;
	for(int i = 0;i < n;++i) res1 ^= 1LL * (i + 1) * (nxt[i] + 1);
	for(int i = 0;i < m;++i) res2 ^= 1LL * (i + 1) * (ext[i] + 1);
	printf("%lld\n%lld\n",res1,res2);
	return 0;
}
```



### manacher

```c++
#include<bits/stdc++.h>
using namespace std;
#define maxn 12000000
char c[maxn<<1],s[maxn<<1];
int p[maxn<<1];
int cnt,n,mr,ans,mid;
void manacher()
{
    s[++cnt]='~'; s[++cnt]='#';
    for(int i=1;i<=n;i++) s[++cnt]=c[i],s[++cnt]='#';
    s[++cnt]='!';
    for(int i=2;i<cnt;i++){
        if(i<=mr) p[i]=min(p[mid*2-i],mr-i+1);
        else p[i]=1;
        while(s[i-p[i]]==s[i+p[i]]) p[i]++;
        if(i+p[i]>mr) mr=i+p[i]-1,mid=i;
        ans=max(ans,p[i]);
    }
}
int main(){
    cin>>c+1; n=strlen(c+1);
    manacher();
    cout<<ans-1<<endl; 
    return 0;
}
```

### trie树

```c++
#include<bits/stdc++.h>
using namespace std;
int n,m;
#define maxn 500050
char a[55];
struct Trie{
	int ch[maxn][26],sz,val[maxn];
	Trie(){
		sz=1; memset(ch[0],0,sizeof(ch[0])); memset(val,0,sizeof(val));
	}
	int idx(char c){return c-'a';}
	void insert(char *s){
		int u=0,len=strlen(s+1);
		for(int i=1;i<=len;i++){
			int c=idx(s[i]);
			if(!ch[u][c]){
				memset(ch[sz],0,sizeof(ch[sz]));
				ch[u][c]=sz++;
			}
			u=ch[u][c];
		}
		//val[u]==xx 对叶子节点进行操作 
	}
	int search(char *s){
		int u=0,len=strlen(s+1);
		for(int i=1;i<=len;i++){
			int c=idx(s[i]);
			if(!ch[u][c]) return 1;
			u=ch[u][c];
		}
		if(!val[u]){
			val[u]=1; return 0;
		}
		else return 2;
	}
}tree;	
int main(){
cin>>n; for(int i=1;i<=n;i++) cin>>a+1,tree.insert(a);
cin>>m;
for(int i=1;i<=m;i++){
	cin>>a+1; int x=tree.search(a);
	if(x==1) cout<<"WRONG"<<endl;
	if(x==2) cout<<"REPEAT"<<endl;
	if(x==0) cout<<"OK"<<endl;
}
	return 0;
}
```

### ac自动机

```c++
/*
 * @Author: CSU_WJJJ 
 * @Date: 2021-05-08 23:50:29 
 * @Last Modified by: CSU_WJJJ
 * @Last Modified time: 2021-05-09 00:12:32
 */
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define maxn 500010
int n;
char s[1000050];
struct tree{
    int ch[maxn][26],val[maxn],fail[maxn],cnt;
    queue<int>q;
    int idx(char c){return c-'a';}
    void insert(char *s)
    {
        int len=strlen(s+1),u=0;
        for(int i=1;i<=len;i++){
            int c=idx(s[i]);
            if(!ch[u][c]){
                // memset(ch[cnt],0,sizeof(ch[cnt]));
                ch[u][c]=++cnt;
            }
            u=ch[u][c];
        }
        val[u]++;
    }
    void build()
    {
        for(int i=0;i<26;i++) if(ch[0][i]) fail[ch[0][i]]=0,q.push(ch[0][i]);
        while(!q.empty()){
            int u=q.front(); q.pop();
            for(int i=0;i<26;i++){
                if(ch[u][i]) fail[ch[u][i]]=ch[fail[u]][i],q.push(ch[u][i]);
                else ch[u][i]=ch[fail[u]][i];
            }
        } 
    }
    int query(char *s)
    {
        int len=strlen(s+1),u=0,ans=0;
        for(int i=1;i<=len;i++){
            int tmp=idx(s[i]);
            u=ch[u][tmp];
            for(int j=u;j&&val[j]!=-1;j=fail[j]){
                ans+=val[j]; val[j]=-1;
            }
        }
        return ans;
    }
}ac;
int main(){
cin>>n;
for(int i=1;i<=n;i++) cin>>s+1,ac.insert(s);
ac.build();
cin>>s+1; cout<<ac.query(s)<<endl;
    return 0;
}
```



## 数据结构

### 并查集

```cpp
int find(int x){
	return x==fa[x]?fa[x]:find(fa[x]);
}
```

```cpp
int find(int x){
	if(x==fa[x]) return x;
    int rt=find(fa[x]);
    d[x]+=d[fa[x]];
    return fa[x]=rt;
}
//带权并查集 先递归 回溯的时候维护父子关系
```

### 树状数组

```cpp
int lowbit(int x)
{
    return x&-x;
}
int ask(int x)
{
	int ans=0;
    for(;x;x-=lowbit(x)) ans+=c[x];
    return ans;
}
void add(int x,int y)
{
    for(;x<=n;x+=lowbit(x)) c[x]+=y;
}
// 求逆序对
ll ans=0;
for(int i=1;i<=n;i++){
    add(q[i],1); ans+=i-ask(q[i]);
}
cout<<ans<<endl;
```

### 单调队列

```cpp
cin>>n>>k;
for(int i=1;i<=n;i++){
	cin>>a[i];
}
int head=1,tail=0;
for(int i=1;i<=n;i++){
	while(head<=tail&&que[tail]>=a[i]) tail--;
	que[++tail]=a[i]; pos[tail]=i;
	while(pos[head]<=i-k) head++;
	if(i>=k){
		cout<<que[head]<<endl;
	}
}
```

### 线段树

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define IOS ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
#define maxn 100050
int a[maxn];
struct tree{
    int l,r;
    ll val,lazy;
}t[maxn<<2];
int n,m;
void build(int k,int l,int r)
{
    if(l==r){
        t[k]={l,r,a[l],0}; return;
    }
    else t[k]={l,r,0,0};
    int mid=l+r>>1;
    build(k<<1,l,mid); build(k<<1|1,mid+1,r);
    t[k].val=t[k<<1].val+t[k<<1|1].val;
}
void pushdown(int k)
{
    if(t[k].lazy){
        t[k<<1].val+=t[k].lazy*(t[k<<1].r-t[k<<1].l+1);
        t[k<<1|1].val+=t[k].lazy*(t[k<<1|1].r-t[k<<1|1].l+1);
        t[k<<1|1].lazy+=t[k].lazy;
        t[k<<1].lazy+=t[k].lazy;
        t[k].lazy=0;
    }
}
void change(int k,int l,int r,ll z)
{
    if(t[k].l>=l&&t[k].r<=r){
        t[k].val+=z*(t[k].r-t[k].l+1); t[k].lazy+=z;
        return;
    }
    pushdown(k);
    int mid=t[k].l+t[k].r>>1;
    if(l<=mid) change(k<<1,l,r,z);
    if(r>mid) change(k<<1|1,l,r,z);
    t[k].val=t[k<<1].val+t[k<<1|1].val;
}
ll query(int k,int l,int r)
{
    ll ans=0;
    if(t[k].l>=l&&t[k].r<=r) return t[k].val;
    pushdown(k);
    int mid=t[k].l+t[k].r>>1;
    if(l<=mid) ans+=query(k<<1,l,r);
    if(r>mid) ans+=query(k<<1|1,l,r);
    return ans;
}
int main(){
IOS
cin>>n>>m;
for(int i=1;i<=n;i++){
    cin>>a[i];
}
build(1,1,n);
for(int i=1;i<=m;i++){
    int x; cin>>x;
    if(x==1){
        int a,b,c; cin>>a>>b>>c;
        change(1,a,b,c);
    }
    else{
        int a,b; cin>>a>>b;
        cout<<query(1,a,b)<<endl;
    }
}
    return 0;
}
```

### 扫描线

给n个矩形 让求出n个矩形的面积并

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define inf 0x3f3f3f3f
#define endl '\n'
int n;
ll x,xx,y,yy;
vector<int>vx;
struct scanline{
    ll l,r,h;
    int mark;
    bool operator < (const scanline &a) const{
        return h<a.h;
    }
};
struct tree
{
    int l,r,sum;
    ll len;
}t[1000050<<2];
vector<scanline>vs;
void build(int k,int l,int r)
{
    t[k]={l,r,0,0};
    if(l==r) return;
    int mid=l+r>>1;
    build(k<<1,l,mid); build(k<<1|1,mid+1,r);
}
void pushup(int k)
{
    int l=t[k].l,r=t[k].r;
    if(t[k].sum) t[k].len=vx[r]-vx[l-1];
    else{
        t[k].len=t[k<<1].len+t[k<<1|1].len;
    }
}
void change(int k,ll l,ll r,int c)
{
    // cout<<vx[t[k].r]<<" "<<l<<" "<<r<<" "<<vx[t[k].l-1]<<endl;
    if(vx[t[k].r]<=l||r<=vx[t[k].l-1]) return;
    if(l<=vx[t[k].l-1]&&vx[t[k].r]<=r){
        t[k].sum+=c; pushup(k);
        return;
    }
    change(k<<1,l,r,c); change(k<<1|1,l,r,c);
    pushup(k);
}
int main(){
ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
cin>>n;
for(int i=1;i<=n;i++){
    cin>>x>>y>>xx>>yy;
    vx.push_back(x); vx.push_back(xx);
    vs.push_back({x,xx,y,1}); vs.push_back({x,xx,yy,-1});
}
sort(vx.begin(),vx.end()); sort(vs.begin(),vs.end());
int tot=unique(vx.begin(),vx.end())-vx.begin();
// cout<<tot<<endl;
build(1,1,tot-1);
ll ans=0;
// for(auto x:vx) cout<<x<<endl;
for(int i=0;i<vs.size()-1;i++){
    change(1,vs[i].l,vs[i].r,vs[i].mark);
    // cout<<vs[i].l<<" "<<vs[i].r<<" "<<vs[i].mark<<endl;
    ans+=t[1].len*(vs[i+1].h-vs[i].h);
}
cout<<ans<<endl;
    return 0;
}
```



### 主席树

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
int n,m;
const int N = 200050;
int rt[N*20],ls[N*20],rs[N*20],cnt,sum[N*20];
int a[N];
vector<int>alls;
int findx(int x)
{
    return lower_bound(alls.begin(),alls.end(),x)-alls.begin()+1;
}
void build(int &t,int l,int r)
{
    t=++cnt;
    if(l==r) return;
    int mid=l+r>>1;
    build(ls[t],l,mid); build(rs[t],mid+1,r);
}
void insert(int &t,int pre,int l,int r,int x)
{
    t=++cnt;
    ls[t]=ls[pre]; rs[t]=rs[pre]; sum[t]=sum[pre]+1;
    if(l==r) return;
    int mid=l+r>>1;
    if(x<=mid) insert(ls[t],ls[pre],l,mid,x);
    if(x>mid) insert(rs[t],rs[pre],mid+1,r,x);
}
int query(int u,int v,int l,int r,int x)
{
    if(l==r) return l;
    int tmp=sum[ls[v]]-sum[ls[u]];
    int mid=l+r>>1;
    if(x<=tmp) return query(ls[u],ls[v],l,mid,x);
    else return query(rs[u],rs[v],mid+1,r,x-tmp);
}
int main(){
ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
cin>>n>>m;
for(int i=1;i<=n;i++) cin>>a[i],alls.push_back(a[i]);
sort(alls.begin(),alls.end());
alls.erase(unique(alls.begin(),alls.end()),alls.end());
int q=alls.size();
build(rt[0],1,q);
for(int i=1;i<=n;i++){
    int x=findx(a[i]);
    insert(rt[i],rt[i-1],1,q,x);
}
for(int i=1;i<=m;i++){
    int aa,b,c; cin>>aa>>b>>c;
    // cout<<query(rt[aa-1],rt[b],1,q,c)<<endl;
    cout<<alls[query(rt[aa-1],rt[b],1,q,c)-1]<<endl;
}
    return 0;
}
```

### 平衡树

```c++
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define IOS ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
const int maxn=100050;
struct node{
    int l,r,size,key,val;
}t[maxn];
int n,cnt,root;
int New_node(int x)
{
    t[++cnt].size=1;
    t[cnt].val=x;
    t[cnt].key=rand();
    return cnt;
}
void Update(int x)
{
    t[x].size=t[t[x].l].size+t[t[x].r].size+1;
}
void Split(int now,int val,int &x,int &y)
{
    if(!now) x=y=0;
    else{
        if(t[now].val<=val){
            x=now; Split(t[now].r,val,t[now].r,y);
        }
        else{
            y=now; Split(t[now].l,val,x,t[now].l);
        }
        Update(now);
    }
}
int Merge(int x,int y)
{
    if(!x||!y) return x+y;
    if(t[x].key>t[y].key){
        t[x].r=Merge(t[x].r,y);
        Update(x); return x;
    }
    else{
        t[y].l=Merge(x,t[y].l);
        Update(y); return y;
    }
}
void Insert(int val)
{
    int x,y; Split(root,val,x,y);
    root=Merge(Merge(x,New_node(val)),y);
}
void Delete(int val)
{
    int x,y,z;
    Split(root,val,x,z); Split(x,val-1,x,y);
    y=Merge(t[y].l,t[y].r); root=Merge(Merge(x,y),z);
}
int Find_rank_of_x(int val)
{
    int x,y; Split(root,val-1,x,y);
    cout<<t[x].size+1<<endl;
    int tmp=t[x].size+1;
    root=Merge(x,y);
    return tmp;
}
void Find_xth(int val)
{
    int now=root;
    while(now){
        if(t[t[now].l].size+1==val) break;
        if(t[t[now].l].size>=val) now=t[now].l;
        else{
            val-=t[t[now].l].size+1;
            now=t[now].r;
        }
    }
    cout<<t[now].val<<endl;
}
void Find_pre(int val)
{
    int x,y; Split(root,val-1,x,y);
    int now=x;
    while(t[now].r) now=t[now].r;
    cout<<t[now].val<<endl;
    root=Merge(x,y);
}
void Find_aft(int val)
{
    int x,y; Split(root,val,x,y);
    int now=y;
    while(t[now].l) now=t[now].l;
    cout<<t[now].val<<endl;
    root=Merge(x,y);
}
int main(){
IOS
srand(233);
cin>>n;
for(int i=1;i<=n;i++){
    int x,y; cin>>x>>y;
    if(x==1){
        Insert(y);
    }
    if(x==2){
        Delete(y);
    }
    if(x==3){
        Find_rank_of_x(y);
    }
    if(x==4){
        Find_xth(y);
    }
    if(x==5){
        Find_pre(y);
    }
    if(x==6){
        Find_aft(y);
    }
}
    return 0;
}
```





### 树链剖分

已经看不懂当时的板子了orz

```cpp
struct edge{
	int to,nxt;
}e[M];
struct tree{
	int l,r,sum,maxm;
}t[N<<2];
int num,cnt,n,q;
int f[N],dep[N],siz[N],son[N],head[N];
int top[N],id[N],rk[N],w[N];
void add(int u,int v)
{
	e[++cnt].nxt=head[u];
	e[cnt].to=v;
	head[u]=cnt;
}
void dfs1(int u,int fa)
{
	f[u]=fa,dep[u]=dep[fa]+1,siz[u]=1;
	for(int i=head[u];i;i=e[i].nxt)
	{
		int v=e[i].to;
		if(v!=fa){
			dfs1(v,u);
			siz[u]+=siz[v];
			if(siz[son[u]]<siz[v]){
				son[u]=v;
			}
		}
	}
}
void dfs2(int p,int t)
{
	top[p]=t,id[p]=++num,rk[num]=p;
	if(son[p]) dfs2(son[p],t);
	for(int i=head[p];i;i=e[i].nxt){
		int v=e[i].to;
		if(v!=f[p]&&v!=son[p]){
			dfs2(v,v);
		}
	}
}
void pushup(int k)
{
	t[k].sum=t[ls].sum+t[rs].sum;
	t[k].maxm=max(t[ls].maxm,t[rs].maxm);
}
void build(int k,int l,int r)
{
	t[k].l=l; t[k].r=r;
	if(l==r){
		t[k].sum=t[k].maxm=w[rk[l]];
		return;
	}
	int mid=(l+r)>>1;
	build(ls,l,mid); build(rs,mid+1,r);
	pushup(k);
}
void change(int k,int d,int x)
{
	if(t[k].l==t[k].r&&t[k].r==d){
		t[k].maxm=x;
		t[k].sum=x;
		return;
	}
	int mid=(t[k].l+t[k].r)>>1;
	if(d<=mid){
		change(k<<1,d,x);
	}
	else{
		change(k<<1|1,d,x);
	}
	pushup(k);
}
int Querysum(int k,int l,int r)
{
	int ans=0;
	if(t[k].l>=l&&t[k].r<=r){
		return t[k].sum;
	}
	int mid=(t[k].l+t[k].r)>>1;
	if(l<=mid){
		ans+=Querysum(ls,l,r);
	}
	if(r>mid){
		ans+=Querysum(rs,l,r);
	}
	return ans;
}
int Querymax(int k,int l,int r)
{
	int res=-inf;
	if(t[k].l>=l&&t[k].r<=r){
		return t[k].maxm;
	}
	int mid=(t[k].l+t[k].r)>>1;
	if(l<=mid){
		res=max(res,Querymax(ls,l,r));
	}
	if(r>mid){
		res=max(res,Querymax(rs,l,r));
	}
	return res;
}
int querysum(int x,int y)
{
	int res=0;
	while(top[x]!=top[y])
	{
		if(dep[top[x]]<dep[top[y]]){
			swap(x,y);
		}
		res+=Querysum(1,id[top[x]],id[x]);
		x=f[top[x]];
	}
	if(dep[x]>dep[y]){
		swap(x,y);
	}
	res+=Querysum(1,id[x],id[y]);
	return res;
}
int querymax(int x,int y)
{
	int res=-inf;
	while(top[x]!=top[y])
	{
		if(dep[top[x]]<dep[top[y]]){
			swap(x,y);
		}
		res=max(res,Querymax(1,id[top[x]],id[x]));
		x=f[top[x]];
	}
	if(dep[x]>dep[y]){
		swap(x,y);
	}
	return res=max(res,Querymax(1,id[x],id[y]));
}
int main(){
ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
cin>>n;
for(int i=1;i<=n-1;i++){
	int x,y;
	cin>>x>>y;
	add(x,y); add(y,x);
}
for(int i=1;i<=n;i++){
	cin>>w[i];
}
dfs1(1,1);
dfs2(1,1);
build(1,1,n);
cin>>q;
for(int i=1;i<=q;i++){
	string s;
	int x,y;
	cin>>s>>x>>y;
	if(s[1]=='S'){
		cout<<querysum(x,y)<<endl;
	}
	else if(s[1]=='M'){
		cout<<querymax(x,y)<<endl;
	}
	else if(s[1]=='H'){ 
		change(1,id[x],y);
	}
}
	return 0;
}

```

### 树上启发式合并

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define inf 0x3f3f3f3f
#define endl '\n'
struct edge{
    int nxt,to;
}e[2000050];
int ans[1000050],mx,anss;
int head[1000050],ct,sz[1000050],son[1000050],dep[1000050],cnt[1000050];
void add(int u,int v)
{
    e[++ct].to=v;
    e[ct].nxt=head[u];
    head[u]=ct;
}
void dfs1(int u,int fa)
{
    sz[u]=1; dep[u]=dep[fa]+1;
    for(int i=head[u];i;i=e[i].nxt){
        int v=e[i].to;
        if(v!=fa){
            dfs1(v,u); sz[u]+=sz[v];
            if(sz[v]>sz[son[u]]){
                son[u]=v;
            }
        }
    }
}
void change(int u,int fa,int val,int p)
{
    cnt[dep[u]]+=val;
    if(cnt[dep[u]]>mx){
        mx=cnt[dep[u]]; anss=dep[u];
    }
    else if(cnt[dep[u]]==mx){
        anss=min(anss,dep[u]);
    }
    for(int i=head[u];i;i=e[i].nxt){
        int v=e[i].to;
        if(v==fa||v==p){
            continue;
        }
        change(v,u,val,p);
    }
}
void dfs2(int u,int fa,int op)
{
    for(int i=head[u];i;i=e[i].nxt){
        int v=e[i].to;
        if(v==fa||v==son[u]) continue;
        dfs2(v,u,0);
    }
    if(son[u]) dfs2(son[u],u,1);
    change(u,fa,1,son[u]);
    ans[u]=anss;
    if(op==0){
        change(u,fa,-1,0); mx=anss=0;
    }
}
int main(){
ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
int n;cin>>n;
for(int i=1;i<n;i++){
    int x,y; cin>>x>>y;
    add(x,y); add(y,x);
}
dfs1(1,0); dfs2(1,0,0);
for(int i=1;i<=n;i++){
    cout<<ans[i]-dep[i]<<endl;
}
    return 0;
}
```

## 杂项

### 莫队

```cpp
#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define inf 0x3f3f3f3f
#define endl '\n'
int n,m,a[2000050],pos[2000050],ans,cnt[2000050];
struct node{
    int l,r,id,anss;
}q[2000050];
bool cmp(node a,node b)
{
    return (pos[a.l]^pos[b.l])?pos[a.l]<pos[b.l]:a.r<b.r;
}
void add(int x)
{
    if(!cnt[a[x]]) ans++;
    cnt[a[x]]++;
}
void del(int x)
{
    cnt[a[x]]--;
    if(!cnt[a[x]]) ans--;
}
int main(){
ios::sync_with_stdio(0),cin.tie(0),cout.tie(0);
cin>>n; for(int i=1;i<=n;i++) cin>>a[i];
cin>>m;
for(int i=1;i<=m;i++){
    int x,y; cin>>q[i].l>>q[i].r; q[i].id=i;
}
int block=sqrt(n);
for(int i=1;i<=n;i++) pos[i]=(i-1)/block+1;
sort(q+1,q+1+m,cmp);
int l=1,r=0;
for(int i=1;i<=m;i++){
    while(l<q[i].l) del(l++);
    while(l>q[i].l) add(--l);
    while(r>q[i].r) del(r--);
    while(r<q[i].r) add(++r);
    q[q[i].id].anss=ans;
}
for(int i=1;i<=m;i++){
    cout<<q[i].anss<<endl;
}
    return 0;
}
```