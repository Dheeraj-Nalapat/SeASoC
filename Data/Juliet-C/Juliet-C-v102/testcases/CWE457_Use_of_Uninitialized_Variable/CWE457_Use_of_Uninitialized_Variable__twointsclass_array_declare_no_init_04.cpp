/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE457_Use_of_Uninitialized_Variable__twointsclass_array_declare_no_init_04.cpp
Label Definition File: CWE457_Use_of_Uninitialized_Variable__twointsclass_array.label.xml
Template File: sources-sinks-04.tmpl.cpp
*/
/*
 * @description
 * CWE: 457 Use of Uninitialized Variable
 * BadSource: no_init Don't initialize data
 * GoodSource: Initialize data
 * Sinks: use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data
 * Flow Variant: 04 Control flow: if(static_const_t) and if(static_const_f)
 *
 * */

#include "std_testcase.h"

/* The two variables below are declared "const", so a tool should
   be able to identify that reads of these will always return their
   initialized values. */
static const int static_const_t = 1; /* true */
static const int static_const_f = 0; /* false */

namespace CWE457_Use_of_Uninitialized_Variable__twointsclass_array_declare_no_init_04
{

#ifndef OMITBAD

void bad()
{
    twointsclass * data;
    twointsclass data_uninit_array[10];
    data = data_uninit_array;
    if(static_const_t)
    {
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
    }
    if(static_const_t)
    {
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_const_t to static_const_f */
static void goodB2G1()
{
    twointsclass * data;
    twointsclass data_uninit_array[10];
    data = data_uninit_array;
    if(static_const_t)
    {
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
    }
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
    else
    {
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    twointsclass * data;
    twointsclass data_uninit_array[10];
    data = data_uninit_array;
    if(static_const_t)
    {
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
    }
    if(static_const_t)
    {
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_const_t to static_const_f */
static void goodG2B1()
{
    twointsclass * data;
    twointsclass data_uninit_array[10];
    data = data_uninit_array;
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
    }
    if(static_const_t)
    {
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    twointsclass * data;
    twointsclass data_uninit_array[10];
    data = data_uninit_array;
    if(static_const_t)
    {
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Don't initialize data */
        ; /* empty statement needed for some flow variants */
    }
    if(static_const_t)
    {
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        for(int i=0; i<10; i++)
        {
            data[i].a = i;
            data[i].b = i;
        }
        for(int i=0; i<10; i++)
        {
            printIntLine(data[i].a);
            printIntLine(data[i].b);
        }
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

using namespace CWE457_Use_of_Uninitialized_Variable__twointsclass_array_declare_no_init_04; // so that we can use good and bad easily

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
