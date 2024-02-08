/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE459_Incomplete_Cleanup__wchar_t_07.c
Label Definition File: CWE459_Incomplete_Cleanup.label.xml
Template File: point-flaw-07.tmpl.c
*/
/*
 * @description
 * CWE: 459 Incomplete Cleanup
 * Sinks:
 *    GoodSink: Clean up properly
 *    BadSink : Don't unlink
 * Flow Variant: 07 Control flow: if(static_five==5) and if(static_five!=5)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define UNLINK _wunlink
# define FOPEN _wfopen
#else
# define UNLINK wunlink
# define FOPEN fopen
#endif

/* The variable below is not declared "const", but is never assigned
   any other value so a tool should be able to identify that reads of
   this will always give its initialized value. */
static int static_five = 5;

#ifndef OMITBAD

void CWE459_Incomplete_Cleanup__wchar_t_07_bad()
{
    if(static_five==5)
    {
        {
            FILE *f;
            f = FOPEN(L"temp.txt", L"w");
            if (f != NULL)
            {
                fwprintf(f, L"Temporary file");
                fclose(f);
                /* FLAW: We don't unlink */
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            FILE *f;
            f = FOPEN(L"temp.txt", L"w");
            if (f != NULL)
            {
                fwprintf(f, L"Temporary file");
                fclose(f);
                /* FIX: Unlink the temporary file */
                UNLINK(L"temp.txt"); /* EXPECTED INCIDENTAL: CWE367 TOCTOU - This POSIX API is essentially insecure by design */
            }
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
            FILE *f;
            f = FOPEN(L"temp.txt", L"w");
            if (f != NULL)
            {
                fwprintf(f, L"Temporary file");
                fclose(f);
                /* FLAW: We don't unlink */
            }
        }
    }
    else
    {
        {
            FILE *f;
            f = FOPEN(L"temp.txt", L"w");
            if (f != NULL)
            {
                fwprintf(f, L"Temporary file");
                fclose(f);
                /* FIX: Unlink the temporary file */
                UNLINK(L"temp.txt"); /* EXPECTED INCIDENTAL: CWE367 TOCTOU - This POSIX API is essentially insecure by design */
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(static_five==5)
    {
        {
            FILE *f;
            f = FOPEN(L"temp.txt", L"w");
            if (f != NULL)
            {
                fwprintf(f, L"Temporary file");
                fclose(f);
                /* FIX: Unlink the temporary file */
                UNLINK(L"temp.txt"); /* EXPECTED INCIDENTAL: CWE367 TOCTOU - This POSIX API is essentially insecure by design */
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            FILE *f;
            f = FOPEN(L"temp.txt", L"w");
            if (f != NULL)
            {
                fwprintf(f, L"Temporary file");
                fclose(f);
                /* FLAW: We don't unlink */
            }
        }
    }
}

void CWE459_Incomplete_Cleanup__wchar_t_07_good()
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
    CWE459_Incomplete_Cleanup__wchar_t_07_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE459_Incomplete_Cleanup__wchar_t_07_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
