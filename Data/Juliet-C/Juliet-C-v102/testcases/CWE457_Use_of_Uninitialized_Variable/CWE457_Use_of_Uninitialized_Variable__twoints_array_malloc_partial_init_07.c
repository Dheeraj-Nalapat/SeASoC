/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_partial_init_07.c
Label Definition File: CWE457_Use_of_Uninitialized_Variable.c_array.label.xml
Template File: sources-sinks-07.tmpl.c
*/
/*
 * @description
 * CWE: 457 Use of Uninitialized Variable
 * BadSource: partial_init Initialize part, but not all of the array
 * GoodSource: Initialize data
 * Sinks: use
 *    GoodSink: Initialize then use data
 *    BadSink : Use data
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_partial_init_07_bad()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<(10/2); i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_five==5 to static_five!=5 */
static void goodB2G1()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<(10/2); i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
    else
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<(10/2); i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_five==5 to static_five!=5 */
static void goodG2B1()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<(10/2); i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    else
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    twoints * data;
    data = (twoints *)malloc(10*sizeof(twoints));
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<(10/2); i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
    }
    if(static_five==5)
    {
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            int i;
            for(i=0; i<10; i++)
            {
                data[i].a = i;
                data[i].b = i;
            }
        }
        {
            int i;
            for(i=0; i<10; i++)
            {
                printIntLine(data[i].a);
                printIntLine(data[i].b);
            }
        }
    }
}

void CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_partial_init_07_good()
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
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_partial_init_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE457_Use_of_Uninitialized_Variable__twoints_array_malloc_partial_init_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
