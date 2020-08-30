//filename: add.c
#include <stdio.h>

struct CObj {
    int f;
    int f2;
};

struct CObjRet {
    int o;
    int r;
};

int add(int a, int b)
{
    return a + b;
}

struct CObjRet add_obj(struct CObj *obj) {
    struct CObjRet o;
    o.r = obj->f + obj->f2;
    o.o = obj->f;
    return o;
}

typedef void (*pfnAddObjCallback)(struct CObj *a, struct CObjRet *ret);

struct CObjRet AddObjCallback(pfnAddObjCallback callback, struct CObj a)
{
    printf("ok %d %d\n", a.f, a.f2);
    struct CObjRet r;
    callback(&a, &r);
    return r;
}
