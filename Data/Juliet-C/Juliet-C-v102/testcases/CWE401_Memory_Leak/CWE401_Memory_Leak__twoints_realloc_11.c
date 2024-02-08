/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__twoints_realloc_11.c
Label Definition File: CWE401_Memory_Leak.c.label.xml
Template File: sources-sinks-11.tmpl.c
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource: realloc Allocate data using realloc()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 11 Control flow: if(global_returns_t()) and if(global_returns_f())
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE401_Memory_Leak__twoints_realloc_11_bad()
{
    twoints * data;
    data = NULL;
    if(global_returns_t())
    {
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = (twoints *)realloc(data, 100*sizeof(twoints));
        /* Initialize and make use of data */
        data[0].a = 0;
        data[0].b = 0;
        printStructLine(&data[0]);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Use memory allocated on the stack */
            twoints data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0].a = 0;
            data[0].b = 0;
            printStructLine(&data[0]);
        }
    }
    if(global_returns_t())
    {
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Deallocate memory */
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second global_returns_t() to global_returns_f() */
static void goodB2G1()
{
    twoints * data;
    data = NULL;
    if(global_returns_t())
    {
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = (twoints *)realloc(data, 100*sizeof(twoints));
        /* Initialize and make use of data */
        data[0].a = 0;
        data[0].b = 0;
        printStructLine(&data[0]);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Use memory allocated on the stack */
            twoints data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0].a = 0;
            data[0].b = 0;
            printStructLine(&data[0]);
        }
    }
    if(global_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* FIX: Deallocate memory */
        free(data);
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    twoints * data;
    data = NULL;
    if(global_returns_t())
    {
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = (twoints *)realloc(data, 100*sizeof(twoints));
        /* Initialize and make use of data */
        data[0].a = 0;
        data[0].b = 0;
        printStructLine(&data[0]);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Use memory allocated on the stack */
            twoints data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0].a = 0;
            data[0].b = 0;
            printStructLine(&data[0]);
        }
    }
    if(global_returns_t())
    {
        /* FIX: Deallocate memory */
        free(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first global_returns_t() to global_returns_f() */
static void goodG2B1()
{
    twoints * data;
    data = NULL;
    if(global_returns_f())
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = (twoints *)realloc(data, 100*sizeof(twoints));
        /* Initialize and make use of data */
        data[0].a = 0;
        data[0].b = 0;
        printStructLine(&data[0]);
    }
    else
    {
        {
            /* FIX: Use memory allocated on the stack */
            twoints data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0].a = 0;
            data[0].b = 0;
            printStructLine(&data[0]);
        }
    }
    if(global_returns_t())
    {
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Deallocate memory */
        free(data);
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    twoints * data;
    data = NULL;
    if(global_returns_t())
    {
        {
            /* FIX: Use memory allocated on the stack */
            twoints data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0].a = 0;
            data[0].b = 0;
            printStructLine(&data[0]);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = NULL;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = (twoints *)realloc(data, 100*sizeof(twoints));
        /* Initialize and make use of data */
        data[0].a = 0;
        data[0].b = 0;
        printStructLine(&data[0]);
    }
    if(global_returns_t())
    {
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Deallocate memory */
        free(data);
    }
}

void CWE401_Memory_Leak__twoints_realloc_11_good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE401_Memory_Leak__twoints_realloc_11_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE401_Memory_Leak__twoints_realloc_11_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
