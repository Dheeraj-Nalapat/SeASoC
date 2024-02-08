/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__malloc_realloc_long_long_07.c
Label Definition File: CWE401_Memory_Leak__malloc_realloc.label.xml
Template File: point-flaw-07.tmpl.c
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * Sinks:
 *    GoodSink: Ensure the memory block pointed to by data is always freed
 *    BadSink : malloc() and use then realloc() and use data before free()
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE401_Memory_Leak__malloc_realloc_long_long_07_bad()
{
    if(static_five==5)
    {
        {
            long long * data = (long long *)malloc(100*sizeof(long long));
            /* Initialize and make use of data */
            data[0] = 5L;
            printLongLongLine(data[0]);
            /* FLAW: If realloc() fails, the initial memory block will not be freed() */
            data = (long long *)realloc(data, (130000)*sizeof(long long));
            if (data != NULL)
            {
                /* Reinitialize and make use of data */
                data[0] = 10L;
                printLongLongLine(data[0]);
                free(data);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            long long * data = (long long *)malloc(100*sizeof(long long));
            long long * tmp;
            /* Initialize and make use of data */
            data[0] = 5L;
            printLongLongLine(data[0]);
            tmp = (long long *)realloc(data, (130000)*sizeof(long long));
            /* FIX: Ensure realloc() was successful before assigning data to the memory block
             * allocated with realloc() */
            if (tmp != NULL)
            {
                data = tmp;
                /* Reinitialize and make use of data */
                data[0] = 10L;
                printLongLongLine(data[0]);
            }
            free(data);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(static_five!=5) instead of if(static_five==5) */
static void good1()
{
    if(static_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            long long * data = (long long *)malloc(100*sizeof(long long));
            /* Initialize and make use of data */
            data[0] = 5L;
            printLongLongLine(data[0]);
            /* FLAW: If realloc() fails, the initial memory block will not be freed() */
            data = (long long *)realloc(data, (130000)*sizeof(long long));
            if (data != NULL)
            {
                /* Reinitialize and make use of data */
                data[0] = 10L;
                printLongLongLine(data[0]);
                free(data);
            }
        }
    }
    else
    {
        {
            long long * data = (long long *)malloc(100*sizeof(long long));
            long long * tmp;
            /* Initialize and make use of data */
            data[0] = 5L;
            printLongLongLine(data[0]);
            tmp = (long long *)realloc(data, (130000)*sizeof(long long));
            /* FIX: Ensure realloc() was successful before assigning data to the memory block
             * allocated with realloc() */
            if (tmp != NULL)
            {
                data = tmp;
                /* Reinitialize and make use of data */
                data[0] = 10L;
                printLongLongLine(data[0]);
            }
            free(data);
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_five==5)
    {
        {
            long long * data = (long long *)malloc(100*sizeof(long long));
            long long * tmp;
            /* Initialize and make use of data */
            data[0] = 5L;
            printLongLongLine(data[0]);
            tmp = (long long *)realloc(data, (130000)*sizeof(long long));
            /* FIX: Ensure realloc() was successful before assigning data to the memory block
             * allocated with realloc() */
            if (tmp != NULL)
            {
                data = tmp;
                /* Reinitialize and make use of data */
                data[0] = 10L;
                printLongLongLine(data[0]);
            }
            free(data);
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            long long * data = (long long *)malloc(100*sizeof(long long));
            /* Initialize and make use of data */
            data[0] = 5L;
            printLongLongLine(data[0]);
            /* FLAW: If realloc() fails, the initial memory block will not be freed() */
            data = (long long *)realloc(data, (130000)*sizeof(long long));
            if (data != NULL)
            {
                /* Reinitialize and make use of data */
                data[0] = 10L;
                printLongLongLine(data[0]);
                free(data);
            }
        }
    }
}

void CWE401_Memory_Leak__malloc_realloc_long_long_07_good()
{
    good1();
    good2();
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
    CWE401_Memory_Leak__malloc_realloc_long_long_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE401_Memory_Leak__malloc_realloc_long_long_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
