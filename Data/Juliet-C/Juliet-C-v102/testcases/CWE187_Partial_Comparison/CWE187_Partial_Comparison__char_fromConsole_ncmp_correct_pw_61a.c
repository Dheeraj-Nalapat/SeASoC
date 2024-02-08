/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61a.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-61a.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: fromConsole Read input from the console
 * GoodSource: Provide a matching password
 * Sinks: ncmp_correct_pw
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use strncmp() to do password match, but use the length of the correct password
 * Flow Variant: 61 Data flow: data returned from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD "Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ strlen(PASSWORD)

#ifndef OMITBAD

/* bad function declaration */
char * CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61b_bad_source(char * data);

void CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61b_bad_source(data);
    /* By using the strlen() of the correct password, you are able to create a partial comparison */
    /* For example if PASSWORD=PASSWORD1234, a user supplied password of PASSWORD123456 will allow access */
    /* POTENTIAL FLAW: Possibly partially compare the two passwords */
    if (strncmp(PASSWORD, data, strlen(PASSWORD)) == 0)
        printLine("Access granted");
    else
        printLine("Access denied!");
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
char * CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61b_goodG2B_source(char * data);

static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61b_goodG2B_source(data);
    /* By using the strlen() of the correct password, you are able to create a partial comparison */
    /* For example if PASSWORD=PASSWORD1234, a user supplied password of PASSWORD123456 will allow access */
    /* POTENTIAL FLAW: Possibly partially compare the two passwords */
    if (strncmp(PASSWORD, data, strlen(PASSWORD)) == 0)
        printLine("Access granted");
    else
        printLine("Access denied!");
}

/* goodB2G uses the BadSource with the GoodSink */
char * CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61b_goodB2G_source(char * data);

static void goodB2G()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    data = CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61b_goodB2G_source(data);
    /* Ideally, we would want to do a check to see if the passwords are of equal length */
    /* FIX: Compare the two passwords completely and correctly */
    if (strcmp(PASSWORD, data) == 0)
        printLine("Access granted");
    else
        printLine("Access denied!");
}

void CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61_good()
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
    CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__char_fromConsole_ncmp_correct_pw_61_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
