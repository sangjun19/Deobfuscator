	.file	"madhus2003r_SSCD-18CSL66_3_flatten.c"
	.text
	.globl	input
	.bss
	.align 8
	.type	input, @object
	.size	input, 10
input:
	.zero	10
	.globl	_TIG_IZ_gBAw_envp
	.align 8
	.type	_TIG_IZ_gBAw_envp, @object
	.size	_TIG_IZ_gBAw_envp, 8
_TIG_IZ_gBAw_envp:
	.zero	8
	.globl	stack
	.align 16
	.type	stack, @object
	.size	stack, 25
stack:
	.zero	25
	.globl	curp
	.align 16
	.type	curp, @object
	.size	curp, 20
curp:
	.zero	20
	.globl	_TIG_IZ_gBAw_argv
	.align 8
	.type	_TIG_IZ_gBAw_argv, @object
	.size	_TIG_IZ_gBAw_argv, 8
_TIG_IZ_gBAw_argv:
	.zero	8
	.globl	follow
	.align 16
	.type	follow, @object
	.size	follow, 30
follow:
	.zero	30
	.globl	table
	.align 32
	.type	table, @object
	.size	table, 120
table:
	.zero	120
	.globl	_TIG_IZ_gBAw_argc
	.align 4
	.type	_TIG_IZ_gBAw_argc, @object
	.size	_TIG_IZ_gBAw_argc, 4
_TIG_IZ_gBAw_argc:
	.zero	4
	.globl	first
	.align 16
	.type	first, @object
	.size	first, 30
first:
	.zero	30
	.globl	prod
	.align 16
	.type	prod, @object
	.size	prod, 30
prod:
	.zero	30
	.globl	top
	.align 4
	.type	top, @object
	.size	top, 4
top:
	.zero	4
	.text
	.globl	display
	.type	display, @function
display:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$5, -8(%rbp)
.L11:
	cmpq	$5, -8(%rbp)
	je	.L2
	cmpq	$5, -8(%rbp)
	ja	.L12
	cmpq	$4, -8(%rbp)
	je	.L13
	cmpq	$4, -8(%rbp)
	ja	.L12
	cmpq	$1, -8(%rbp)
	je	.L5
	cmpq	$3, -8(%rbp)
	je	.L6
	jmp	.L12
.L5:
	cmpl	$0, -12(%rbp)
	js	.L8
	movq	$3, -8(%rbp)
	jmp	.L10
.L8:
	movq	$4, -8(%rbp)
	jmp	.L10
.L6:
	movl	-12(%rbp), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	putchar@PLT
	subl	$1, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L10
.L2:
	movl	top(%rip), %eax
	movl	%eax, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L10
.L12:
	nop
.L10:
	jmp	.L11
.L13:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	display, .-display
	.globl	pop
	.type	pop, @function
pop:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	$0, -8(%rbp)
.L19:
	cmpq	$0, -8(%rbp)
	je	.L15
	cmpq	$1, -8(%rbp)
	jne	.L21
	jmp	.L20
.L15:
	movl	top(%rip), %eax
	subl	$1, %eax
	movl	%eax, top(%rip)
	movq	$1, -8(%rbp)
	jmp	.L18
.L21:
	nop
.L18:
	jmp	.L19
.L20:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	pop, .-pop
	.globl	numr
	.type	numr, @function
numr:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$6, -8(%rbp)
.L43:
	cmpq	$7, -8(%rbp)
	ja	.L44
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L25(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L25(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L25:
	.long	.L32-.L25
	.long	.L31-.L25
	.long	.L30-.L25
	.long	.L29-.L25
	.long	.L28-.L25
	.long	.L27-.L25
	.long	.L26-.L25
	.long	.L24-.L25
	.text
.L28:
	movl	$2, %eax
	jmp	.L33
.L31:
	movl	$1, %eax
	jmp	.L33
.L29:
	movq	$1, -8(%rbp)
	jmp	.L34
.L26:
	movsbl	-20(%rbp), %eax
	subl	$64, %eax
	cmpl	$34, %eax
	ja	.L35
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L37(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L37(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L37:
	.long	.L41-.L37
	.long	.L40-.L37
	.long	.L39-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L35-.L37
	.long	.L38-.L37
	.long	.L36-.L37
	.text
.L41:
	movq	$5, -8(%rbp)
	jmp	.L42
.L36:
	movq	$2, -8(%rbp)
	jmp	.L42
.L38:
	movq	$0, -8(%rbp)
	jmp	.L42
.L39:
	movq	$4, -8(%rbp)
	jmp	.L42
.L40:
	movq	$7, -8(%rbp)
	jmp	.L42
.L35:
	movq	$3, -8(%rbp)
	nop
.L42:
	jmp	.L34
.L27:
	movl	$3, %eax
	jmp	.L33
.L32:
	movl	$1, %eax
	jmp	.L33
.L24:
	movl	$1, %eax
	jmp	.L33
.L30:
	movl	$2, %eax
	jmp	.L33
.L44:
	nop
.L34:
	jmp	.L43
.L33:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	numr, .-numr
	.section	.rodata
.LC0:
	.string	"e"
	.align 8
.LC1:
	.string	"Enter the input string terminated with $ to parse:-"
.LC2:
	.string	"%s"
.LC3:
	.string	"Invalid String"
.LC4:
	.string	"Invalid String - Rejected"
	.align 8
.LC5:
	.string	"\n-----------------------------------"
.LC6:
	.string	"\n"
.LC7:
	.string	"Stack\t Input\tAction"
	.align 8
.LC8:
	.string	"\n--------------------------------------------"
.LC9:
	.string	"\nInvalid String - Rejected"
.LC10:
	.string	"\nGrammar"
.LC11:
	.string	"\tApply production %s\n"
	.align 8
.LC12:
	.string	"\n------------------------------------------"
.LC13:
	.string	"\nfirst={%s,%s,%s}"
.LC14:
	.string	"\nfollow={%s,%s}\n"
	.align 8
.LC15:
	.string	"\nPredictive parsing table for the given grammar :"
.LC16:
	.string	"\nValid String - Accepted"
.LC17:
	.string	"\t\t%s\t"
	.align 8
.LC18:
	.string	"\n\nInput String Entered Without End Marker $"
.LC19:
	.string	"%-30s"
.LC20:
	.string	"\tMatched %c\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$112, %rsp
	movl	%edi, -84(%rbp)
	movq	%rsi, -96(%rbp)
	movq	%rdx, -104(%rbp)
	movl	$0, -68(%rbp)
	jmp	.L46
.L47:
	movl	-68(%rbp), %eax
	cltq
	leaq	curp(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -68(%rbp)
.L46:
	cmpl	$19, -68(%rbp)
	jle	.L47
	nop
.L48:
	movl	$0, -64(%rbp)
	jmp	.L49
.L50:
	movl	-64(%rbp), %eax
	cltq
	leaq	stack(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -64(%rbp)
.L49:
	cmpl	$24, -64(%rbp)
	jle	.L50
	nop
.L51:
	movl	$-1, top(%rip)
	nop
.L52:
	movl	$0, -60(%rbp)
	jmp	.L53
.L54:
	movl	-60(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -60(%rbp)
.L53:
	cmpl	$9, -60(%rbp)
	jle	.L54
	nop
.L55:
	movb	$0, table(%rip)
	movb	$0, 1+table(%rip)
	movb	$0, 2+table(%rip)
	movb	$0, 3+table(%rip)
	movb	$0, 4+table(%rip)
	movb	$0, 5+table(%rip)
	movb	$0, 6+table(%rip)
	movb	$0, 7+table(%rip)
	movb	$0, 8+table(%rip)
	movb	$0, 9+table(%rip)
	movb	$0, 10+table(%rip)
	movb	$0, 11+table(%rip)
	movb	$0, 12+table(%rip)
	movb	$0, 13+table(%rip)
	movb	$0, 14+table(%rip)
	movb	$0, 15+table(%rip)
	movb	$0, 16+table(%rip)
	movb	$0, 17+table(%rip)
	movb	$0, 18+table(%rip)
	movb	$0, 19+table(%rip)
	movb	$0, 20+table(%rip)
	movb	$0, 21+table(%rip)
	movb	$0, 22+table(%rip)
	movb	$0, 23+table(%rip)
	movb	$0, 24+table(%rip)
	movb	$0, 25+table(%rip)
	movb	$0, 26+table(%rip)
	movb	$0, 27+table(%rip)
	movb	$0, 28+table(%rip)
	movb	$0, 29+table(%rip)
	movb	$0, 30+table(%rip)
	movb	$0, 31+table(%rip)
	movb	$0, 32+table(%rip)
	movb	$0, 33+table(%rip)
	movb	$0, 34+table(%rip)
	movb	$0, 35+table(%rip)
	movb	$0, 36+table(%rip)
	movb	$0, 37+table(%rip)
	movb	$0, 38+table(%rip)
	movb	$0, 39+table(%rip)
	movb	$0, 40+table(%rip)
	movb	$0, 41+table(%rip)
	movb	$0, 42+table(%rip)
	movb	$0, 43+table(%rip)
	movb	$0, 44+table(%rip)
	movb	$0, 45+table(%rip)
	movb	$0, 46+table(%rip)
	movb	$0, 47+table(%rip)
	movb	$0, 48+table(%rip)
	movb	$0, 49+table(%rip)
	movb	$0, 50+table(%rip)
	movb	$0, 51+table(%rip)
	movb	$0, 52+table(%rip)
	movb	$0, 53+table(%rip)
	movb	$0, 54+table(%rip)
	movb	$0, 55+table(%rip)
	movb	$0, 56+table(%rip)
	movb	$0, 57+table(%rip)
	movb	$0, 58+table(%rip)
	movb	$0, 59+table(%rip)
	movb	$0, 60+table(%rip)
	movb	$0, 61+table(%rip)
	movb	$0, 62+table(%rip)
	movb	$0, 63+table(%rip)
	movb	$0, 64+table(%rip)
	movb	$0, 65+table(%rip)
	movb	$0, 66+table(%rip)
	movb	$0, 67+table(%rip)
	movb	$0, 68+table(%rip)
	movb	$0, 69+table(%rip)
	movb	$0, 70+table(%rip)
	movb	$0, 71+table(%rip)
	movb	$0, 72+table(%rip)
	movb	$0, 73+table(%rip)
	movb	$0, 74+table(%rip)
	movb	$0, 75+table(%rip)
	movb	$0, 76+table(%rip)
	movb	$0, 77+table(%rip)
	movb	$0, 78+table(%rip)
	movb	$0, 79+table(%rip)
	movb	$0, 80+table(%rip)
	movb	$0, 81+table(%rip)
	movb	$0, 82+table(%rip)
	movb	$0, 83+table(%rip)
	movb	$0, 84+table(%rip)
	movb	$0, 85+table(%rip)
	movb	$0, 86+table(%rip)
	movb	$0, 87+table(%rip)
	movb	$0, 88+table(%rip)
	movb	$0, 89+table(%rip)
	movb	$0, 90+table(%rip)
	movb	$0, 91+table(%rip)
	movb	$0, 92+table(%rip)
	movb	$0, 93+table(%rip)
	movb	$0, 94+table(%rip)
	movb	$0, 95+table(%rip)
	movb	$0, 96+table(%rip)
	movb	$0, 97+table(%rip)
	movb	$0, 98+table(%rip)
	movb	$0, 99+table(%rip)
	movb	$0, 100+table(%rip)
	movb	$0, 101+table(%rip)
	movb	$0, 102+table(%rip)
	movb	$0, 103+table(%rip)
	movb	$0, 104+table(%rip)
	movb	$0, 105+table(%rip)
	movb	$0, 106+table(%rip)
	movb	$0, 107+table(%rip)
	movb	$0, 108+table(%rip)
	movb	$0, 109+table(%rip)
	movb	$0, 110+table(%rip)
	movb	$0, 111+table(%rip)
	movb	$0, 112+table(%rip)
	movb	$0, 113+table(%rip)
	movb	$0, 114+table(%rip)
	movb	$0, 115+table(%rip)
	movb	$0, 116+table(%rip)
	movb	$0, 117+table(%rip)
	movb	$0, 118+table(%rip)
	movb	$0, 119+table(%rip)
	nop
.L56:
	movb	$36, follow(%rip)
	movb	$0, 1+follow(%rip)
	movb	$97, 10+follow(%rip)
	movb	$0, 11+follow(%rip)
	movb	$97, 20+follow(%rip)
	movb	$0, 21+follow(%rip)
	nop
.L57:
	movb	$97, first(%rip)
	movb	$0, 1+first(%rip)
	movb	$98, 10+first(%rip)
	movb	$0, 11+first(%rip)
	movb	$64, 20+first(%rip)
	movb	$0, 21+first(%rip)
	nop
.L58:
	movb	$65, prod(%rip)
	movb	$45, 1+prod(%rip)
	movb	$62, 2+prod(%rip)
	movb	$97, 3+prod(%rip)
	movb	$66, 4+prod(%rip)
	movb	$97, 5+prod(%rip)
	movb	$0, 6+prod(%rip)
	movb	$66, 10+prod(%rip)
	movb	$45, 11+prod(%rip)
	movb	$62, 12+prod(%rip)
	movb	$98, 13+prod(%rip)
	movb	$66, 14+prod(%rip)
	movb	$0, 15+prod(%rip)
	movb	$66, 20+prod(%rip)
	movb	$45, 21+prod(%rip)
	movb	$62, 22+prod(%rip)
	movb	$64, 23+prod(%rip)
	movb	$0, 24+prod(%rip)
	nop
.L59:
	movq	$0, _TIG_IZ_gBAw_envp(%rip)
	nop
.L60:
	movq	$0, _TIG_IZ_gBAw_argv(%rip)
	nop
.L61:
	movl	$0, _TIG_IZ_gBAw_argc(%rip)
	nop
	nop
.L62:
.L63:
#APP
# 369 "madhus2003r_SSCD-18CSL66_3.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-gBAw--0
# 0 "" 2
#NO_APP
	movl	-84(%rbp), %eax
	movl	%eax, _TIG_IZ_gBAw_argc(%rip)
	movq	-96(%rbp), %rax
	movq	%rax, _TIG_IZ_gBAw_argv(%rip)
	movq	-104(%rbp), %rax
	movq	%rax, _TIG_IZ_gBAw_envp(%rip)
	nop
	movq	$71, -16(%rbp)
.L171:
	cmpq	$91, -16(%rbp)
	ja	.L173
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L66(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L66(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L66:
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L122-.L66
	.long	.L121-.L66
	.long	.L173-.L66
	.long	.L120-.L66
	.long	.L119-.L66
	.long	.L118-.L66
	.long	.L117-.L66
	.long	.L116-.L66
	.long	.L115-.L66
	.long	.L114-.L66
	.long	.L113-.L66
	.long	.L112-.L66
	.long	.L111-.L66
	.long	.L110-.L66
	.long	.L109-.L66
	.long	.L108-.L66
	.long	.L107-.L66
	.long	.L106-.L66
	.long	.L173-.L66
	.long	.L105-.L66
	.long	.L173-.L66
	.long	.L104-.L66
	.long	.L103-.L66
	.long	.L102-.L66
	.long	.L101-.L66
	.long	.L100-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L99-.L66
	.long	.L98-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L97-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L96-.L66
	.long	.L95-.L66
	.long	.L94-.L66
	.long	.L173-.L66
	.long	.L93-.L66
	.long	.L92-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L91-.L66
	.long	.L90-.L66
	.long	.L173-.L66
	.long	.L89-.L66
	.long	.L173-.L66
	.long	.L88-.L66
	.long	.L87-.L66
	.long	.L86-.L66
	.long	.L85-.L66
	.long	.L84-.L66
	.long	.L83-.L66
	.long	.L173-.L66
	.long	.L82-.L66
	.long	.L173-.L66
	.long	.L81-.L66
	.long	.L80-.L66
	.long	.L173-.L66
	.long	.L79-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L78-.L66
	.long	.L77-.L66
	.long	.L173-.L66
	.long	.L76-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L75-.L66
	.long	.L173-.L66
	.long	.L74-.L66
	.long	.L173-.L66
	.long	.L73-.L66
	.long	.L72-.L66
	.long	.L71-.L66
	.long	.L173-.L66
	.long	.L70-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L69-.L66
	.long	.L68-.L66
	.long	.L173-.L66
	.long	.L173-.L66
	.long	.L67-.L66
	.long	.L65-.L66
	.text
.L107:
	addl	$1, -56(%rbp)
	movq	$77, -16(%rbp)
	jmp	.L123
.L72:
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	numr
	movl	%eax, -44(%rbp)
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	numr
	movl	%eax, -40(%rbp)
	movl	-40(%rbp), %eax
	cltq
	movq	%rax, %rdx
	salq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	movq	%rax, %rdx
	movl	-44(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	leaq	table(%rip), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	curp(%rip), %rax
	movq	%rax, %rdi
	call	strcpy@PLT
	leaq	.LC0(%rip), %rax
	movq	%rax, %rsi
	leaq	curp(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -48(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L123
.L102:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	input(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	$0, -56(%rbp)
	movq	$51, -16(%rbp)
	jmp	.L123
.L89:
	cmpl	$2, -56(%rbp)
	jg	.L124
	movq	$37, -16(%rbp)
	jmp	.L123
.L124:
	movq	$19, -16(%rbp)
	jmp	.L123
.L87:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %edi
	call	exit@PLT
.L99:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$36, %al
	jne	.L126
	movq	$58, -16(%rbp)
	jmp	.L123
.L126:
	movq	$63, -16(%rbp)
	jmp	.L123
.L111:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$97, %al
	je	.L128
	movq	$61, -16(%rbp)
	jmp	.L123
.L128:
	movq	$17, -16(%rbp)
	jmp	.L123
.L110:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L123
.L83:
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$36, %al
	jne	.L130
	movq	$30, -16(%rbp)
	jmp	.L123
.L130:
	movq	$15, -16(%rbp)
	jmp	.L123
.L73:
	cmpl	$2, -52(%rbp)
	jle	.L132
	movq	$6, -16(%rbp)
	jmp	.L123
.L132:
	movq	$53, -16(%rbp)
	jmp	.L123
.L98:
	cmpl	$3, -52(%rbp)
	jg	.L134
	movq	$7, -16(%rbp)
	jmp	.L123
.L134:
	movq	$21, -16(%rbp)
	jmp	.L123
.L113:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$60, -16(%rbp)
	jmp	.L123
.L77:
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %edx
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rcx
	movzbl	(%rax,%rcx), %eax
	cmpb	%al, %dl
	jne	.L136
	movq	$86, -16(%rbp)
	jmp	.L123
.L136:
	movq	$68, -16(%rbp)
	jmp	.L123
.L117:
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	prod(%rip), %rdx
	addq	%rdx, %rax
	movq	%rax, %rdi
	call	puts@PLT
	addl	$1, -56(%rbp)
	movq	$34, -16(%rbp)
	jmp	.L123
.L85:
	addl	$1, -56(%rbp)
	movq	$55, -16(%rbp)
	jmp	.L123
.L71:
	movzbl	3+curp(%rip), %eax
	cmpb	$64, %al
	jne	.L138
	movq	$11, -16(%rbp)
	jmp	.L123
.L138:
	movq	$9, -16(%rbp)
	jmp	.L123
.L104:
	cmpl	$3, -52(%rbp)
	jne	.L140
	movq	$12, -16(%rbp)
	jmp	.L123
.L140:
	movq	$60, -16(%rbp)
	jmp	.L123
.L74:
	cmpl	$2, -56(%rbp)
	jg	.L142
	movq	$83, -16(%rbp)
	jmp	.L123
.L142:
	movq	$25, -16(%rbp)
	jmp	.L123
.L121:
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	movq	%rax, %rdx
	leaq	prod(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	numr
	movl	%eax, -24(%rbp)
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	movq	%rax, %rdx
	leaq	first(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	numr
	movl	%eax, -20(%rbp)
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	prod(%rip), %rdx
	leaq	(%rax,%rdx), %rcx
	movl	-20(%rbp), %eax
	cltq
	movq	%rax, %rdx
	salq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	movq	%rax, %rdx
	movl	-24(%rbp), %eax
	movslq	%eax, %rsi
	movq	%rsi, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	leaq	table(%rip), %rax
	addq	%rdx, %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$54, -16(%rbp)
	jmp	.L123
.L109:
	cmpl	$0, -48(%rbp)
	je	.L144
	movq	$90, -16(%rbp)
	jmp	.L123
.L144:
	movq	$13, -16(%rbp)
	jmp	.L123
.L103:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$36, %al
	je	.L146
	movq	$52, -16(%rbp)
	jmp	.L123
.L146:
	movq	$17, -16(%rbp)
	jmp	.L123
.L105:
	addl	$1, -56(%rbp)
	movq	$49, -16(%rbp)
	jmp	.L123
.L78:
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$64, %al
	jle	.L148
	movq	$87, -16(%rbp)
	jmp	.L123
.L148:
	movq	$53, -16(%rbp)
	jmp	.L123
.L101:
	movl	$36, %edi
	call	push
	movl	$65, %edi
	call	push
	movl	$0, -56(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$53, -16(%rbp)
	jmp	.L123
.L114:
	call	pop
	movq	$53, -16(%rbp)
	jmp	.L123
.L116:
	call	pop
	leaq	curp(%rip), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	subl	$1, %eax
	movl	%eax, -52(%rbp)
	movq	$79, -16(%rbp)
	jmp	.L123
.L112:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L79:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L123
.L88:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	testb	%al, %al
	je	.L150
	movq	$14, -16(%rbp)
	jmp	.L123
.L150:
	movq	$39, -16(%rbp)
	jmp	.L123
.L106:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -56(%rbp)
	movq	$34, -16(%rbp)
	jmp	.L123
.L108:
	addl	$1, -56(%rbp)
	movq	$51, -16(%rbp)
	jmp	.L123
.L67:
	leaq	curp(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$81, -16(%rbp)
	jmp	.L123
.L84:
	cmpl	$2, -56(%rbp)
	jg	.L152
	movq	$42, -16(%rbp)
	jmp	.L123
.L152:
	movq	$27, -16(%rbp)
	jmp	.L123
.L81:
	addl	$1, -52(%rbp)
	movq	$91, -16(%rbp)
	jmp	.L123
.L119:
	movl	-52(%rbp), %eax
	cltq
	leaq	curp(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	push
	subl	$1, -52(%rbp)
	movq	$79, -16(%rbp)
	jmp	.L123
.L100:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, -56(%rbp)
	movq	$77, -16(%rbp)
	jmp	.L123
.L95:
	leaq	20+first(%rip), %rax
	movq	%rax, %rcx
	leaq	10+first(%rip), %rax
	movq	%rax, %rdx
	leaq	first(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	10+follow(%rip), %rax
	movq	%rax, %rdx
	leaq	follow(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movb	$0, table(%rip)
	movw	$97, 10+table(%rip)
	movw	$98, 20+table(%rip)
	movw	$36, 30+table(%rip)
	movw	$65, 40+table(%rip)
	movw	$66, 80+table(%rip)
	movl	$0, -56(%rbp)
	movq	$55, -16(%rbp)
	jmp	.L123
.L80:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$98, %al
	je	.L154
	movq	$24, -16(%rbp)
	jmp	.L123
.L154:
	movq	$17, -16(%rbp)
	jmp	.L123
.L68:
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$91, %al
	jg	.L156
	movq	$80, -16(%rbp)
	jmp	.L123
.L156:
	movq	$53, -16(%rbp)
	jmp	.L123
.L82:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -16(%rbp)
	jmp	.L123
.L97:
	cmpl	$2, -56(%rbp)
	jg	.L158
	movq	$8, -16(%rbp)
	jmp	.L123
.L158:
	movq	$38, -16(%rbp)
	jmp	.L123
.L75:
	call	display
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$56, -16(%rbp)
	jmp	.L123
.L76:
	movl	$0, -56(%rbp)
	movq	$49, -16(%rbp)
	jmp	.L123
.L86:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$36, %al
	je	.L160
	movq	$10, -16(%rbp)
	jmp	.L123
.L160:
	movq	$75, -16(%rbp)
	jmp	.L123
.L90:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$0, %edi
	call	exit@PLT
.L120:
	call	display
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$69, -16(%rbp)
	jmp	.L123
.L65:
	cmpl	$3, -52(%rbp)
	jg	.L162
	movq	$46, -16(%rbp)
	jmp	.L123
.L162:
	movq	$18, -16(%rbp)
	jmp	.L123
.L96:
	movl	$0, -52(%rbp)
	movq	$31, -16(%rbp)
	jmp	.L123
.L93:
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	movq	%rax, %rdx
	leaq	prod(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	numr
	movl	%eax, -36(%rbp)
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	movq	%rax, %rdx
	leaq	follow(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	movsbl	%al, %eax
	movl	%eax, %edi
	call	numr
	movl	%eax, -32(%rbp)
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	leaq	prod(%rip), %rdx
	leaq	(%rax,%rdx), %rcx
	movl	-32(%rbp), %eax
	cltq
	movq	%rax, %rdx
	salq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	movq	%rax, %rdx
	movl	-36(%rbp), %eax
	movslq	%eax, %rsi
	movq	%rsi, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	leaq	table(%rip), %rax
	addq	%rdx, %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movq	$54, -16(%rbp)
	jmp	.L123
.L115:
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$36, %al
	je	.L164
	movq	$5, -16(%rbp)
	jmp	.L123
.L164:
	movq	$75, -16(%rbp)
	jmp	.L123
.L92:
	movl	-56(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	addq	%rax, %rax
	movq	%rax, %rdx
	leaq	first(%rip), %rax
	movzbl	(%rdx,%rax), %eax
	cmpb	$64, %al
	je	.L166
	movq	$3, -16(%rbp)
	jmp	.L123
.L166:
	movq	$41, -16(%rbp)
	jmp	.L123
.L91:
	movl	-52(%rbp), %eax
	cltq
	movq	%rax, %rdx
	salq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	movq	%rax, %rdx
	movl	-56(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	leaq	table(%rip), %rax
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -16(%rbp)
	jmp	.L123
.L94:
	movl	-56(%rbp), %eax
	subl	$1, %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	cmpb	$36, %al
	je	.L168
	movq	$47, -16(%rbp)
	jmp	.L123
.L168:
	movq	$26, -16(%rbp)
	jmp	.L123
.L70:
	movl	$0, -52(%rbp)
	movq	$91, -16(%rbp)
	jmp	.L123
.L118:
	movl	-52(%rbp), %eax
	cltq
	movq	%rax, %rdx
	salq	$2, %rdx
	addq	%rax, %rdx
	leaq	(%rdx,%rdx), %rax
	movq	%rax, %rdx
	movl	-56(%rbp), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	salq	$2, %rax
	addq	%rcx, %rax
	salq	$3, %rax
	addq	%rax, %rdx
	leaq	table(%rip), %rax
	addq	%rdx, %rax
	movl	$1414548805, (%rax)
	movw	$89, 4(%rax)
	addl	$1, -52(%rbp)
	movq	$31, -16(%rbp)
	jmp	.L123
.L69:
	movl	-56(%rbp), %eax
	cltq
	leaq	input(%rip), %rdx
	movzbl	(%rax,%rdx), %eax
	movsbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	pop
	addl	$1, -56(%rbp)
	movq	$53, -16(%rbp)
	jmp	.L123
.L122:
	movl	$0, %eax
	jmp	.L172
.L173:
	nop
.L123:
	jmp	.L171
.L172:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.globl	push
	.type	push, @function
push:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	%edi, %eax
	movb	%al, -20(%rbp)
	movq	$0, -8(%rbp)
.L180:
	cmpq	$2, -8(%rbp)
	je	.L175
	cmpq	$2, -8(%rbp)
	ja	.L182
	cmpq	$0, -8(%rbp)
	je	.L177
	cmpq	$1, -8(%rbp)
	jne	.L182
	jmp	.L181
.L177:
	movq	$2, -8(%rbp)
	jmp	.L179
.L175:
	movl	top(%rip), %eax
	addl	$1, %eax
	movl	%eax, top(%rip)
	movl	top(%rip), %eax
	cltq
	leaq	stack(%rip), %rcx
	movzbl	-20(%rbp), %edx
	movb	%dl, (%rax,%rcx)
	movq	$1, -8(%rbp)
	jmp	.L179
.L182:
	nop
.L179:
	jmp	.L180
.L181:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	push, .-push
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
