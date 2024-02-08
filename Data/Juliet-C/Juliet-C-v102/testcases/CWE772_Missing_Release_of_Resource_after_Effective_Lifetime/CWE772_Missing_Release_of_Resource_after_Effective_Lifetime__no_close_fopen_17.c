/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__no_close_fopen_17.c
Label Definition File: CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__no_close.label.xml
Template File: source-sinks-17.tmpl.c
*/
/*
 * @description
 * CWE: 772 Missing Release of Resource after Effective Lifetime
 * BadSource: fopen Open a file using fopen()
 * Sinks:
 *    GoodSink: Close the file using fclose()
 *    BadSink : Do not close file
 * Flow Variant: 17 Control flow: for loops
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__no_close_fopen_17_bad()
{
    int j,k;
    FILE * data;
    data = NULL;
    data = fopen("BadSource_fopen.txt", "w+");
    for(j = 0; j < 1; j++)
    {
        /* FLAW: No attempt to close the file */
        ; /* empty statement needed for some flow variants */
    }
    for(k = 0; k < 0; k++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: If the file is still opened, close it */
        if (data != NULL)
        {
            fclose(data);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G() - use badsource and goodsink by changing the conditions on the for statements */
static void goodB2G()
{
    int j,k;
    FILE * data;
    data = NULL;
    data = fopen("BadSource_fopen.txt", "w+");
    for(j = 0; j < 0; j++)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FLAW: No attempt to close the file */
        ; /* empty statement needed for some flow variants */
    }
    for(k = 0; k < 1; k++)
    {
        /* FIX: If the file is still opened, close it */
        if (data != NULL)
        {
            fclose(data);
        }
    }
}

void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__no_close_fopen_17_good()
{
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
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__no_close_fopen_17_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__no_close_fopen_17_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
