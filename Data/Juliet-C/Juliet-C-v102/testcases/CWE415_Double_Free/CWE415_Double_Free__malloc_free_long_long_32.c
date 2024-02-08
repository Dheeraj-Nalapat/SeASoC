/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE415_Double_Free__malloc_free_long_long_32.c
Label Definition File: CWE415_Double_Free__malloc_free.label.xml
Template File: sources-sinks-32.tmpl.c
*/
/*
 * @description
 * CWE: 415 Double Free
 * BadSource:  Allocate data using malloc() and Deallocate data using free()
 * GoodSource: Allocate data using malloc()
 * Sinks:
 *    GoodSink: do nothing
 *    BadSink : Deallocate data using free()
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE415_Double_Free__malloc_free_long_long_32_bad()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data */
    data = NULL;
    {
        long long * data = *data_ptr1;
        data = (long long *)malloc(100*sizeof(long long));
        /* POTENTIAL FLAW: Free data in the source - the bad sink frees data as well */
        free(data);
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* POTENTIAL FLAW: Possibly freeing memory twice */
        free(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data */
    data = NULL;
    {
        long long * data = *data_ptr1;
        data = (long long *)malloc(100*sizeof(long long));
        /* FIX: Do NOT free data in the source - the bad sink frees data */
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* POTENTIAL FLAW: Possibly freeing memory twice */
        free(data);
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data */
    data = NULL;
    {
        long long * data = *data_ptr1;
        data = (long long *)malloc(100*sizeof(long long));
        /* POTENTIAL FLAW: Free data in the source - the bad sink frees data as well */
        free(data);
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* do nothing */
        /* FIX: Don't attempt to free the memory */
        ; /* empty statement needed for some flow variants */
    }
}

void CWE415_Double_Free__malloc_free_long_long_32_good()
{
    goodG2B();
    goodB2G();
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
    CWE415_Double_Free__malloc_free_long_long_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE415_Double_Free__malloc_free_long_long_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
