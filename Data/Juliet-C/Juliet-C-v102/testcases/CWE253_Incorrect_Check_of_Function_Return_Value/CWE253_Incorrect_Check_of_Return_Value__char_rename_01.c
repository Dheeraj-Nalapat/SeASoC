/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE253_Incorrect_Check_of_Return_Value__char_rename_01.c
Label Definition File: CWE253_Incorrect_Check_of_Return_Value.string.label.xml
Template File: point-flaw-01.tmpl.c
*/
/*
 * @description
 * CWE: 253 Incorrect Check of Return Value
 * Sinks: rename
 *    GoodSink: Correctly check if rename() failed
 *    BadSink : Incorrectlycheck if rename() failed
 * Flow Variant: 01 Baseline
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define RENAME rename
#else
# define RENAME rename
#endif

#define OLD_BAD_FILE_NAME "oldbadfilename.txt"
#define OLD_GOOD_FILE_NAME "oldgoodfilename.txt"

#ifndef OMITBAD

void CWE253_Incorrect_Check_of_Return_Value__char_rename_01_bad()
{
    {
        /* FLAW: rename() might fail, in which case the return value will be non-zero, but
         * we are checking to see if the return value is 0 */
        if (RENAME(OLD_BAD_FILE_NAME, "newbadfilename.txt") == 0)
        {
            printLine("rename failed!");
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

static void good1()
{
    {
        /* FIX: check for the correct return value */
        if (RENAME(OLD_GOOD_FILE_NAME, "newgoodfilename.txt") != 0)
        {
            printLine("rename failed!");
        }
    }
}

void CWE253_Incorrect_Check_of_Return_Value__char_rename_01_good()
{
    good1();
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
    CWE253_Incorrect_Check_of_Return_Value__char_rename_01_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE253_Incorrect_Check_of_Return_Value__char_rename_01_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
