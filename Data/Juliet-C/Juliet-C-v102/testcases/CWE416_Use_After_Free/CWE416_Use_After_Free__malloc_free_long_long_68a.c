/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE416_Use_After_Free__malloc_free_long_long_68a.c
Label Definition File: CWE416_Use_After_Free__malloc_free.label.xml
Template File: sources-sinks-68a.tmpl.c
*/
/*
 * @description
 * CWE: 416 Use After Free
 * BadSource:  Allocate data using malloc(), initialize memory block, and Deallocate data using free()
 * GoodSource: Allocate data using malloc() and initialize memory block
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Use data
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

long long * CWE416_Use_After_Free__malloc_free_long_long_68_bad_data;
long long * CWE416_Use_After_Free__malloc_free_long_long_68_goodG2B_data;
long long * CWE416_Use_After_Free__malloc_free_long_long_68_goodB2G_data;

#ifndef OMITBAD

/* bad function declaration */
void CWE416_Use_After_Free__malloc_free_long_long_68b_bad_sink();

void CWE416_Use_After_Free__malloc_free_long_long_68_bad()
{
    long long * data;
    /* Initialize data */
    data = NULL;
    data = (long long *)malloc(100*sizeof(long long));
    {
        size_t i;
        for(i = 0; i < 100; i++)
        {
            data[i] = 5L;
        }
    }
    /* POTENTIAL FLAW: Free data in the source - the bad sink attempts to use data */
    free(data);
    CWE416_Use_After_Free__malloc_free_long_long_68_bad_data = data;
    CWE416_Use_After_Free__malloc_free_long_long_68b_bad_sink();
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declarations */
void CWE416_Use_After_Free__malloc_free_long_long_68b_goodG2B_sink();
void CWE416_Use_After_Free__malloc_free_long_long_68b_goodB2G_sink();

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    long long * data;
    /* Initialize data */
    data = NULL;
    data = (long long *)malloc(100*sizeof(long long));
    {
        size_t i;
        for(i = 0; i < 100; i++)
        {
            data[i] = 5L;
        }
    }
    /* FIX: Do not delete data in the source */
    CWE416_Use_After_Free__malloc_free_long_long_68_goodG2B_data = data;
    CWE416_Use_After_Free__malloc_free_long_long_68b_goodG2B_sink();
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G()
{
    long long * data;
    /* Initialize data */
    data = NULL;
    data = (long long *)malloc(100*sizeof(long long));
    {
        size_t i;
        for(i = 0; i < 100; i++)
        {
            data[i] = 5L;
        }
    }
    /* POTENTIAL FLAW: Free data in the source - the bad sink attempts to use data */
    free(data);
    CWE416_Use_After_Free__malloc_free_long_long_68_goodB2G_data = data;
    CWE416_Use_After_Free__malloc_free_long_long_68b_goodB2G_sink();
}

void CWE416_Use_After_Free__malloc_free_long_long_68_good()
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
    CWE416_Use_After_Free__malloc_free_long_long_68_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE416_Use_After_Free__malloc_free_long_long_68_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
