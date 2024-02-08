/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__new_array_int_05.cpp
Label Definition File: CWE401_Memory_Leak__new_array.label.xml
Template File: sources-sinks-05.tmpl.cpp
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using new[]
 * GoodSource: Point data to a stack buffer
 * Sinks:
 *    GoodSink: call delete[] on data
 *    BadSink : no deallocation of data
 * Flow Variant: 05 Control flow: if(static_t) and if(static_f)
 *
 * */

#include "std_testcase.h"

/* The two variables below are not defined as "const", but are never
   assigned any other value, so a tool should be able to identify that
   reads of these will always return their initialized values. */
static int static_t = 1; /* true */
static int static_f = 0; /* false */

namespace CWE401_Memory_Leak__new_array_int_05
{

#ifndef OMITBAD

void bad()
{
    int * data;
    data = NULL;
    if(static_t)
    {
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new int[100];
        /* Initialize and make use of data */
        data[0] = 5;
        printIntLine(data[0]);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Use memory allocated on the stack */
            int data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0] = 5;
            printIntLine(data[0]);
        }
    }
    if(static_t)
    {
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Deallocate memory */
        delete[] data;
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_t to static_f */
static void goodB2G1()
{
    int * data;
    data = NULL;
    if(static_t)
    {
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new int[100];
        /* Initialize and make use of data */
        data[0] = 5;
        printIntLine(data[0]);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Use memory allocated on the stack */
            int data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0] = 5;
            printIntLine(data[0]);
        }
    }
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* FIX: Deallocate memory */
        delete[] data;
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    int * data;
    data = NULL;
    if(static_t)
    {
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new int[100];
        /* Initialize and make use of data */
        data[0] = 5;
        printIntLine(data[0]);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: Use memory allocated on the stack */
            int data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0] = 5;
            printIntLine(data[0]);
        }
    }
    if(static_t)
    {
        /* FIX: Deallocate memory */
        delete[] data;
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_t to static_f */
static void goodG2B1()
{
    int * data;
    data = NULL;
    if(static_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new int[100];
        /* Initialize and make use of data */
        data[0] = 5;
        printIntLine(data[0]);
    }
    else
    {
        {
            /* FIX: Use memory allocated on the stack */
            int data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0] = 5;
            printIntLine(data[0]);
        }
    }
    if(static_t)
    {
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Deallocate memory */
        delete[] data;
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    int * data;
    data = NULL;
    if(static_t)
    {
        {
            /* FIX: Use memory allocated on the stack */
            int data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize and make use of data */
            data[0] = 5;
            printIntLine(data[0]);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new int[100];
        /* Initialize and make use of data */
        data[0] = 5;
        printIntLine(data[0]);
    }
    if(static_t)
    {
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Deallocate memory */
        delete[] data;
    }
}

void good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE401_Memory_Leak__new_array_int_05; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
