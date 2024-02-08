/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE252_Unchecked_Return_Value__wchar_t_rename_14.c
Label Definition File: CWE252_Unchecked_Return_Value.string.label.xml
Template File: point-flaw-14.tmpl.c
*/
/*
 * @description
 * CWE: 252 Unchecked Return Value
 * Sinks: rename
 *    GoodSink: Check if wrename() fails
 *    BadSink : Do not check if wrename() fails
 * Flow Variant: 14 Control flow: if(global_five==5) and if(global_five!=5)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define RENAME _wrename
#else
# define RENAME wrename
#endif

#define OLD_BAD_FILE_NAME L"oldbadfilename.txt"
#define OLD_GOOD_FILE_NAME L"oldgoodfilename.txt"

#ifndef OMITBAD

void CWE252_Unchecked_Return_Value__wchar_t_rename_14_bad()
{
    if(global_five==5)
    {
        {
            /* FLAW: Do not check the return value */
            RENAME(OLD_BAD_FILE_NAME, L"newbadfilename.txt");
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check the return value */
            if (RENAME(OLD_GOOD_FILE_NAME, L"newgoodfilename.txt") != 0)
            {
                printLine("wrename failed!");
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(global_five!=5) instead of if(global_five==5) */
static void good1()
{
    if(global_five!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Do not check the return value */
            RENAME(OLD_BAD_FILE_NAME, L"newbadfilename.txt");
        }
    }
    else
    {
        {
            /* FIX: check the return value */
            if (RENAME(OLD_GOOD_FILE_NAME, L"newgoodfilename.txt") != 0)
            {
                printLine("wrename failed!");
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(global_five==5)
    {
        {
            /* FIX: check the return value */
            if (RENAME(OLD_GOOD_FILE_NAME, L"newgoodfilename.txt") != 0)
            {
                printLine("wrename failed!");
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: Do not check the return value */
            RENAME(OLD_BAD_FILE_NAME, L"newbadfilename.txt");
        }
    }
}

void CWE252_Unchecked_Return_Value__wchar_t_rename_14_good()
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
    CWE252_Unchecked_Return_Value__wchar_t_rename_14_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE252_Unchecked_Return_Value__wchar_t_rename_14_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
