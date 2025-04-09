	.file	"clean-code-craft-tcq-2_simple-monitor-in-c-ArnoldoZerecero_checker_flatten.c"
	.text
	.globl	printLanguage
	.bss
	.align 4
	.type	printLanguage, @object
	.size	printLanguage, 4
printLanguage:
	.zero	4
	.globl	_TIG_IZ_sFdF_argc
	.align 4
	.type	_TIG_IZ_sFdF_argc, @object
	.size	_TIG_IZ_sFdF_argc, 4
_TIG_IZ_sFdF_argc:
	.zero	4
	.globl	_TIG_IZ_sFdF_envp
	.align 8
	.type	_TIG_IZ_sFdF_envp, @object
	.size	_TIG_IZ_sFdF_envp, 8
_TIG_IZ_sFdF_envp:
	.zero	8
	.globl	_TIG_IZ_sFdF_argv
	.align 8
	.type	_TIG_IZ_sFdF_argv, @object
	.size	_TIG_IZ_sFdF_argv, 8
_TIG_IZ_sFdF_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"Der Ladezustand liegt uber seinem Limit!\nLadezustand: %f\n"
	.align 8
.LC1:
	.string	"Temperature is above its limit!\nTemperature: %f\n"
	.align 8
.LC2:
	.string	"Warnung: Die Temperatur nahert sich ihrem unteren Grenzwert.\nTemperatur: %f\n"
	.align 8
.LC3:
	.string	"Warning: State of Charge is approaching its lower limit.\nState of Charge: %f\n"
	.align 8
.LC4:
	.string	"State of Charge is above its limit!\nState of Charge: %f\n"
	.align 8
.LC5:
	.string	"Warning: Temperature is approaching its upper limit.\nTemperature: %f\n"
	.align 8
.LC6:
	.string	"Warnung: Die Temperatur nahert sich ihrem oberen Grenzwert.\nTemperatur: %f\n"
	.align 8
.LC7:
	.string	"Die Temperatur liegt unter ihrem Grenzwert!\nTemperatur: %f\n"
	.align 8
.LC8:
	.string	"State of Charge is below its limit!\nState of Charge: %f\n"
	.align 8
.LC9:
	.string	"Temperature is below its limit!\nTemperature: %f\n"
	.align 8
.LC10:
	.string	"Warning: State of Charge is approaching its upper limit.\nState of Charge: %f\n"
	.align 8
.LC11:
	.string	"Warnung: Die Laderate nahert sich ihrer Obergrenze.\nLadestrom: %f\n"
	.align 8
.LC12:
	.string	"Warning: Temperature is approaching its lower limit.\nTemperature: %f\n"
	.align 8
.LC13:
	.string	"Warning: Charge Rate is approaching its upper limit.\nCharge Rate: %f\n"
	.align 8
.LC14:
	.string	"Der Ladezustand ist unter seinem Limit!\nLadezustand: %f\n"
	.align 8
.LC15:
	.string	"Warnung: Der Ladezustand nahert sich seiner Obergrenze.\nLadezustand: %f\n"
	.align 8
.LC16:
	.string	"Charge Rate is above its limit!\nCharge Rate: %f\n"
	.align 8
.LC17:
	.string	"Warnung: Der Ladezustand nahert sich seiner unteren Grenze.\nLadezustande: %f\n"
	.align 8
.LC18:
	.string	"Die Temperatur liegt uber ihrem Grenzwert!\nTemperature: %f\n"
	.align 8
.LC19:
	.string	"Die Laderate liegt uber dem Limit!\nLadestrom: %f\n"
	.text
	.globl	printErrorMessage
	.type	printErrorMessage, @function
printErrorMessage:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movss	%xmm0, -24(%rbp)
	movq	$27, -8(%rbp)
.L63:
	movq	-8(%rbp), %rax
	subq	$3, %rax
	cmpq	$43, %rax
	ja	.L64
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L28-.L4
	.long	.L64-.L4
	.long	.L27-.L4
	.long	.L64-.L4
	.long	.L26-.L4
	.long	.L25-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L64-.L4
	.long	.L22-.L4
	.long	.L64-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L64-.L4
	.long	.L16-.L4
	.long	.L64-.L4
	.long	.L15-.L4
	.long	.L64-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L64-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L64-.L4
	.long	.L9-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L65-.L4
	.long	.L64-.L4
	.long	.L7-.L4
	.long	.L64-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L64-.L4
	.long	.L3-.L4
	.text
.L16:
	pxor	%xmm1, %xmm1
	cvtss2sd	-24(%rbp), %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L13:
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L24:
	movl	printLanguage(%rip), %eax
	cmpl	$1, %eax
	jne	.L30
	movq	$15, -8(%rbp)
	jmp	.L29
.L30:
	movq	$39, -8(%rbp)
	jmp	.L29
.L23:
	cmpl	$512, -20(%rbp)
	je	.L32
	cmpl	$512, -20(%rbp)
	jg	.L33
	cmpl	$256, -20(%rbp)
	je	.L34
	cmpl	$256, -20(%rbp)
	jg	.L33
	cmpl	$128, -20(%rbp)
	je	.L35
	cmpl	$128, -20(%rbp)
	jg	.L33
	cmpl	$32, -20(%rbp)
	jg	.L36
	cmpl	$0, -20(%rbp)
	jle	.L33
	cmpl	$32, -20(%rbp)
	ja	.L33
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L38(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L38(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L38:
	.long	.L33-.L38
	.long	.L43-.L38
	.long	.L42-.L38
	.long	.L33-.L38
	.long	.L41-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L40-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L39-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L33-.L38
	.long	.L37-.L38
	.text
.L36:
	cmpl	$64, -20(%rbp)
	je	.L44
	jmp	.L33
.L32:
	movq	$22, -8(%rbp)
	jmp	.L45
.L34:
	movq	$20, -8(%rbp)
	jmp	.L45
.L35:
	movq	$41, -8(%rbp)
	jmp	.L45
.L44:
	movq	$7, -8(%rbp)
	jmp	.L45
.L37:
	movq	$25, -8(%rbp)
	jmp	.L45
.L39:
	movq	$33, -8(%rbp)
	jmp	.L45
.L40:
	movq	$36, -8(%rbp)
	jmp	.L45
.L41:
	movq	$8, -8(%rbp)
	jmp	.L45
.L42:
	movq	$43, -8(%rbp)
	jmp	.L45
.L43:
	movq	$19, -8(%rbp)
	jmp	.L45
.L33:
	movq	$39, -8(%rbp)
	nop
.L45:
	jmp	.L29
.L25:
	pxor	%xmm3, %xmm3
	cvtss2sd	-24(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L17:
	pxor	%xmm4, %xmm4
	cvtss2sd	-24(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L28:
	pxor	%xmm5, %xmm5
	cvtss2sd	-24(%rbp), %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L19:
	pxor	%xmm6, %xmm6
	cvtss2sd	-24(%rbp), %xmm6
	movq	%xmm6, %rax
	movq	%rax, %xmm0
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L9:
	pxor	%xmm7, %xmm7
	cvtss2sd	-24(%rbp), %xmm7
	movq	%xmm7, %rax
	movq	%rax, %xmm0
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L21:
	pxor	%xmm1, %xmm1
	cvtss2sd	-24(%rbp), %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L12:
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L22:
	pxor	%xmm3, %xmm3
	cvtss2sd	-24(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L15:
	movl	printLanguage(%rip), %eax
	testl	%eax, %eax
	jne	.L46
	movq	$29, -8(%rbp)
	jmp	.L29
.L46:
	movq	$14, -8(%rbp)
	jmp	.L29
.L10:
	pxor	%xmm4, %xmm4
	cvtss2sd	-24(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L18:
	pxor	%xmm5, %xmm5
	cvtss2sd	-24(%rbp), %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L5:
	pxor	%xmm6, %xmm6
	cvtss2sd	-24(%rbp), %xmm6
	movq	%xmm6, %rax
	movq	%rax, %xmm0
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L27:
	pxor	%xmm7, %xmm7
	cvtss2sd	-24(%rbp), %xmm7
	movq	%xmm7, %rax
	movq	%rax, %xmm0
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L11:
	pxor	%xmm1, %xmm1
	cvtss2sd	-24(%rbp), %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L7:
	pxor	%xmm2, %xmm2
	cvtss2sd	-24(%rbp), %xmm2
	movq	%xmm2, %rax
	movq	%rax, %xmm0
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L3:
	pxor	%xmm3, %xmm3
	cvtss2sd	-24(%rbp), %xmm3
	movq	%xmm3, %rax
	movq	%rax, %xmm0
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L26:
	pxor	%xmm4, %xmm4
	cvtss2sd	-24(%rbp), %xmm4
	movq	%xmm4, %rax
	movq	%rax, %xmm0
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L14:
	cmpl	$512, -20(%rbp)
	je	.L49
	cmpl	$512, -20(%rbp)
	jg	.L50
	cmpl	$256, -20(%rbp)
	je	.L51
	cmpl	$256, -20(%rbp)
	jg	.L50
	cmpl	$128, -20(%rbp)
	je	.L52
	cmpl	$128, -20(%rbp)
	jg	.L50
	cmpl	$32, -20(%rbp)
	jg	.L53
	cmpl	$0, -20(%rbp)
	jle	.L50
	cmpl	$32, -20(%rbp)
	ja	.L50
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L55(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L55(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L55:
	.long	.L50-.L55
	.long	.L60-.L55
	.long	.L59-.L55
	.long	.L50-.L55
	.long	.L58-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L57-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L56-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L50-.L55
	.long	.L54-.L55
	.text
.L53:
	cmpl	$64, -20(%rbp)
	je	.L61
	jmp	.L50
.L49:
	movq	$5, -8(%rbp)
	jmp	.L62
.L51:
	movq	$46, -8(%rbp)
	jmp	.L62
.L52:
	movq	$34, -8(%rbp)
	jmp	.L62
.L61:
	movq	$23, -8(%rbp)
	jmp	.L62
.L54:
	movq	$3, -8(%rbp)
	jmp	.L62
.L56:
	movq	$32, -8(%rbp)
	jmp	.L62
.L57:
	movq	$21, -8(%rbp)
	jmp	.L62
.L58:
	movq	$44, -8(%rbp)
	jmp	.L62
.L59:
	movq	$30, -8(%rbp)
	jmp	.L62
.L60:
	movq	$17, -8(%rbp)
	jmp	.L62
.L50:
	movq	$39, -8(%rbp)
	nop
.L62:
	jmp	.L29
.L6:
	pxor	%xmm5, %xmm5
	cvtss2sd	-24(%rbp), %xmm5
	movq	%xmm5, %rax
	movq	%rax, %xmm0
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L20:
	pxor	%xmm6, %xmm6
	cvtss2sd	-24(%rbp), %xmm6
	movq	%xmm6, %rax
	movq	%rax, %xmm0
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$1, %eax
	call	printf@PLT
	movq	$39, -8(%rbp)
	jmp	.L29
.L64:
	nop
.L29:
	jmp	.L63
.L65:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	printErrorMessage, .-printErrorMessage
	.globl	stateOfChargeIsInRange
	.type	stateOfChargeIsInRange, @function
stateOfChargeIsInRange:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movq	$7, -8(%rbp)
.L95:
	cmpq	$12, -8(%rbp)
	ja	.L105
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L69(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L69(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L69:
	.long	.L80-.L69
	.long	.L79-.L69
	.long	.L78-.L69
	.long	.L77-.L69
	.long	.L76-.L69
	.long	.L75-.L69
	.long	.L74-.L69
	.long	.L73-.L69
	.long	.L72-.L69
	.long	.L71-.L69
	.long	.L105-.L69
	.long	.L70-.L69
	.long	.L68-.L69
	.text
.L76:
	movl	$128, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L81
.L68:
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L81
.L72:
	movss	.LC20(%rip), %xmm0
	comiss	-20(%rbp), %xmm0
	jbe	.L100
	movq	$6, -8(%rbp)
	jmp	.L81
.L100:
	movq	$3, -8(%rbp)
	jmp	.L81
.L79:
	movl	-20(%rbp), %edx
	movl	-12(%rbp), %eax
	movd	%edx, %xmm0
	movl	%eax, %edi
	call	printErrorMessage
	movq	$5, -8(%rbp)
	jmp	.L81
.L77:
	movss	-20(%rbp), %xmm0
	comiss	.LC21(%rip), %xmm0
	jbe	.L101
	movq	$4, -8(%rbp)
	jmp	.L81
.L101:
	movq	$1, -8(%rbp)
	jmp	.L81
.L70:
	movl	$16, -12(%rbp)
	movl	$0, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L81
.L71:
	movss	.LC22(%rip), %xmm0
	comiss	-20(%rbp), %xmm0
	jbe	.L102
	movq	$11, -8(%rbp)
	jmp	.L81
.L102:
	movq	$2, -8(%rbp)
	jmp	.L81
.L74:
	movl	$64, -12(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L81
.L75:
	movl	-16(%rbp), %eax
	jmp	.L103
.L80:
	movl	$32, -12(%rbp)
	movl	$0, -16(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L81
.L73:
	movq	$12, -8(%rbp)
	jmp	.L81
.L78:
	movss	-20(%rbp), %xmm0
	comiss	.LC23(%rip), %xmm0
	jbe	.L104
	movq	$0, -8(%rbp)
	jmp	.L81
.L104:
	movq	$8, -8(%rbp)
	jmp	.L81
.L105:
	nop
.L81:
	jmp	.L95
.L103:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	stateOfChargeIsInRange, .-stateOfChargeIsInRange
	.globl	chargeRateIsAboveLimit
	.type	chargeRateIsAboveLimit, @function
chargeRateIsAboveLimit:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movq	$4, -8(%rbp)
.L125:
	cmpq	$8, -8(%rbp)
	ja	.L131
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L109(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L109(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L109:
	.long	.L116-.L109
	.long	.L115-.L109
	.long	.L114-.L109
	.long	.L113-.L109
	.long	.L112-.L109
	.long	.L111-.L109
	.long	.L110-.L109
	.long	.L131-.L109
	.long	.L108-.L109
	.text
.L112:
	movq	$2, -8(%rbp)
	jmp	.L117
.L108:
	movl	-16(%rbp), %eax
	jmp	.L128
.L115:
	movss	-20(%rbp), %xmm0
	comiss	.LC24(%rip), %xmm0
	jbe	.L129
	movq	$6, -8(%rbp)
	jmp	.L117
.L129:
	movq	$3, -8(%rbp)
	jmp	.L117
.L113:
	movl	-20(%rbp), %edx
	movl	-12(%rbp), %eax
	movd	%edx, %xmm0
	movl	%eax, %edi
	call	printErrorMessage
	movq	$8, -8(%rbp)
	jmp	.L117
.L110:
	movl	$512, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L117
.L111:
	movl	$256, -12(%rbp)
	movl	$0, -16(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L117
.L116:
	movss	-20(%rbp), %xmm0
	comiss	.LC25(%rip), %xmm0
	jbe	.L130
	movq	$5, -8(%rbp)
	jmp	.L117
.L130:
	movq	$1, -8(%rbp)
	jmp	.L117
.L114:
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L117
.L131:
	nop
.L117:
	jmp	.L125
.L128:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	chargeRateIsAboveLimit, .-chargeRateIsAboveLimit
	.globl	temperatureIsInRange
	.type	temperatureIsInRange, @function
temperatureIsInRange:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movss	%xmm0, -20(%rbp)
	movq	$3, -8(%rbp)
.L161:
	cmpq	$13, -8(%rbp)
	ja	.L171
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L135(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L135(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L135:
	.long	.L171-.L135
	.long	.L146-.L135
	.long	.L145-.L135
	.long	.L144-.L135
	.long	.L143-.L135
	.long	.L142-.L135
	.long	.L141-.L135
	.long	.L140-.L135
	.long	.L139-.L135
	.long	.L138-.L135
	.long	.L137-.L135
	.long	.L136-.L135
	.long	.L171-.L135
	.long	.L134-.L135
	.text
.L143:
	movl	-16(%rbp), %eax
	jmp	.L166
.L139:
	movl	$8, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L148
.L146:
	movl	$1, -12(%rbp)
	movl	$0, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L148
.L144:
	movq	$5, -8(%rbp)
	jmp	.L148
.L136:
	movss	-20(%rbp), %xmm0
	comiss	.LC26(%rip), %xmm0
	jbe	.L167
	movq	$8, -8(%rbp)
	jmp	.L148
.L167:
	movq	$7, -8(%rbp)
	jmp	.L148
.L138:
	movl	$4, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L148
.L134:
	movss	-20(%rbp), %xmm0
	comiss	.LC27(%rip), %xmm0
	jbe	.L168
	movq	$6, -8(%rbp)
	jmp	.L148
.L168:
	movq	$2, -8(%rbp)
	jmp	.L148
.L141:
	movl	$2, -12(%rbp)
	movl	$0, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L148
.L142:
	movl	$1, -16(%rbp)
	movl	$0, -12(%rbp)
	movq	$10, -8(%rbp)
	jmp	.L148
.L137:
	pxor	%xmm0, %xmm0
	comiss	-20(%rbp), %xmm0
	jbe	.L169
	movq	$1, -8(%rbp)
	jmp	.L148
.L169:
	movq	$13, -8(%rbp)
	jmp	.L148
.L140:
	movl	-20(%rbp), %edx
	movl	-12(%rbp), %eax
	movd	%edx, %xmm0
	movl	%eax, %edi
	call	printErrorMessage
	movq	$4, -8(%rbp)
	jmp	.L148
.L145:
	movss	.LC29(%rip), %xmm0
	comiss	-20(%rbp), %xmm0
	jbe	.L170
	movq	$9, -8(%rbp)
	jmp	.L148
.L170:
	movq	$11, -8(%rbp)
	jmp	.L148
.L171:
	nop
.L148:
	jmp	.L161
.L166:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	temperatureIsInRange, .-temperatureIsInRange
	.globl	batteryIsOk
	.type	batteryIsOk, @function
batteryIsOk:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movss	%xmm0, -36(%rbp)
	movss	%xmm1, -40(%rbp)
	movss	%xmm2, -44(%rbp)
	movq	$5, -8(%rbp)
.L192:
	cmpq	$8, -8(%rbp)
	ja	.L194
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L175(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L175(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L175:
	.long	.L183-.L175
	.long	.L182-.L175
	.long	.L181-.L175
	.long	.L180-.L175
	.long	.L179-.L175
	.long	.L178-.L175
	.long	.L177-.L175
	.long	.L176-.L175
	.long	.L174-.L175
	.text
.L179:
	cmpl	$0, -20(%rbp)
	je	.L184
	movq	$8, -8(%rbp)
	jmp	.L186
.L184:
	movq	$3, -8(%rbp)
	jmp	.L186
.L174:
	movl	$1, -24(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L186
.L182:
	movl	$0, -24(%rbp)
	movl	-36(%rbp), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -12(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L186
.L180:
	movl	-24(%rbp), %eax
	jmp	.L193
.L177:
	cmpl	$0, -16(%rbp)
	je	.L188
	movq	$7, -8(%rbp)
	jmp	.L186
.L188:
	movq	$3, -8(%rbp)
	jmp	.L186
.L178:
	movq	$1, -8(%rbp)
	jmp	.L186
.L183:
	cmpl	$0, -12(%rbp)
	je	.L190
	movq	$2, -8(%rbp)
	jmp	.L186
.L190:
	movq	$3, -8(%rbp)
	jmp	.L186
.L176:
	movl	-44(%rbp), %eax
	movd	%eax, %xmm0
	call	chargeRateIsAboveLimit
	movl	%eax, -20(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L186
.L181:
	movl	-40(%rbp), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -16(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L186
.L194:
	nop
.L186:
	jmp	.L192
.L193:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	batteryIsOk, .-batteryIsOk
	.section	.rodata
.LC35:
	.string	"main"
	.align 8
.LC36:
	.string	"clean-code-craft-tcq-2_simple-monitor-in-c-ArnoldoZerecero_checker.c"
.LC37:
	.string	"stateOfChargeIsInRange(50)"
.LC38:
	.string	"temperatureIsInRange(45)"
.LC39:
	.string	"!stateOfChargeIsInRange(19.9)"
.LC43:
	.string	"!batteryIsOk(50, 85, 0)"
.LC44:
	.string	"chargeRateIsAboveLimit(0.7)"
.LC45:
	.string	"chargeRateIsAboveLimit(-1)"
.LC46:
	.string	"stateOfChargeIsInRange(20)"
.LC48:
	.string	"batteryIsOk(1, 21, 0.76)"
.LC49:
	.string	"stateOfChargeIsInRange(20.1)"
.LC50:
	.string	"temperatureIsInRange(0)"
.LC52:
	.string	"temperatureIsInRange(44.9)"
.LC53:
	.string	"stateOfChargeIsInRange(79.9)"
.LC54:
	.string	"batteryIsOk(2.2, 23.9, 0.75)"
.LC55:
	.string	"batteryIsOk(25, 70, 0.7)"
.LC56:
	.string	"!batteryIsOk(44.9, 79.9, 0.9)"
.LC58:
	.string	"!temperatureIsInRange(45.1)"
.LC65:
	.string	"chargeRateIsAboveLimit(0.8)"
.LC66:
	.string	"chargeRateIsAboveLimit(0)"
.LC67:
	.string	"!chargeRateIsAboveLimit(0.9)"
.LC68:
	.string	"stateOfChargeIsInRange(80)"
.LC72:
	.string	"temperatureIsInRange(0.1)"
.LC75:
	.string	"batteryIsOk(44, 79, 0.77)"
.LC78:
	.string	"temperatureIsInRange(22)"
.LC79:
	.string	"!temperatureIsInRange(-0.1)"
.LC80:
	.string	"!stateOfChargeIsInRange(80.1)"
	.text
	.globl	main
	.type	main, @function
main:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$144, %rsp
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movl	$0, printLanguage(%rip)
	nop
.L196:
	movq	$0, _TIG_IZ_sFdF_envp(%rip)
	nop
.L197:
	movq	$0, _TIG_IZ_sFdF_argv(%rip)
	nop
.L198:
	movl	$0, _TIG_IZ_sFdF_argc(%rip)
	nop
	nop
.L199:
.L200:
#APP
# 313 "clean-code-craft-tcq-2_simple-monitor-in-c-ArnoldoZerecero_checker.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-sFdF--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_sFdF_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_sFdF_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_sFdF_envp(%rip)
	nop
	movq	$27, -8(%rbp)
.L331:
	cmpq	$76, -8(%rbp)
	ja	.L333
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L203(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L203(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L203:
	.long	.L278-.L203
	.long	.L277-.L203
	.long	.L276-.L203
	.long	.L275-.L203
	.long	.L274-.L203
	.long	.L273-.L203
	.long	.L272-.L203
	.long	.L271-.L203
	.long	.L270-.L203
	.long	.L269-.L203
	.long	.L268-.L203
	.long	.L267-.L203
	.long	.L266-.L203
	.long	.L265-.L203
	.long	.L264-.L203
	.long	.L263-.L203
	.long	.L262-.L203
	.long	.L261-.L203
	.long	.L260-.L203
	.long	.L259-.L203
	.long	.L258-.L203
	.long	.L257-.L203
	.long	.L256-.L203
	.long	.L255-.L203
	.long	.L254-.L203
	.long	.L253-.L203
	.long	.L252-.L203
	.long	.L251-.L203
	.long	.L250-.L203
	.long	.L249-.L203
	.long	.L248-.L203
	.long	.L247-.L203
	.long	.L246-.L203
	.long	.L245-.L203
	.long	.L244-.L203
	.long	.L243-.L203
	.long	.L242-.L203
	.long	.L241-.L203
	.long	.L240-.L203
	.long	.L333-.L203
	.long	.L239-.L203
	.long	.L238-.L203
	.long	.L237-.L203
	.long	.L236-.L203
	.long	.L235-.L203
	.long	.L234-.L203
	.long	.L233-.L203
	.long	.L232-.L203
	.long	.L231-.L203
	.long	.L230-.L203
	.long	.L229-.L203
	.long	.L228-.L203
	.long	.L227-.L203
	.long	.L226-.L203
	.long	.L225-.L203
	.long	.L224-.L203
	.long	.L223-.L203
	.long	.L222-.L203
	.long	.L221-.L203
	.long	.L220-.L203
	.long	.L219-.L203
	.long	.L218-.L203
	.long	.L217-.L203
	.long	.L216-.L203
	.long	.L215-.L203
	.long	.L214-.L203
	.long	.L213-.L203
	.long	.L212-.L203
	.long	.L211-.L203
	.long	.L210-.L203
	.long	.L209-.L203
	.long	.L208-.L203
	.long	.L207-.L203
	.long	.L206-.L203
	.long	.L205-.L203
	.long	.L204-.L203
	.long	.L202-.L203
	.text
.L260:
	movl	.LC30(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -64(%rbp)
	movq	$67, -8(%rbp)
	jmp	.L279
.L229:
	cmpl	$0, -84(%rbp)
	je	.L280
	movq	$60, -8(%rbp)
	jmp	.L279
.L280:
	movq	$72, -8(%rbp)
	jmp	.L279
.L253:
	cmpl	$0, -72(%rbp)
	je	.L282
	movq	$13, -8(%rbp)
	jmp	.L279
.L282:
	movq	$11, -8(%rbp)
	jmp	.L279
.L230:
	movl	.LC31(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -84(%rbp)
	movq	$50, -8(%rbp)
	jmp	.L279
.L227:
	movss	.LC32(%rip), %xmm2
	movss	.LC33(%rip), %xmm1
	movl	.LC34(%rip), %eax
	movd	%eax, %xmm0
	call	batteryIsOk
	movl	%eax, -24(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L279
.L274:
	cmpl	$0, -24(%rbp)
	je	.L284
	movq	$47, -8(%rbp)
	jmp	.L279
.L284:
	movq	$37, -8(%rbp)
	jmp	.L279
.L248:
	cmpl	$0, -32(%rbp)
	je	.L286
	movq	$10, -8(%rbp)
	jmp	.L279
.L286:
	movq	$32, -8(%rbp)
	jmp	.L279
.L217:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$213, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L264:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$207, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L263:
	movl	.LC25(%rip), %eax
	movd	%eax, %xmm0
	call	chargeRateIsAboveLimit
	movl	%eax, -48(%rbp)
	movq	$61, -8(%rbp)
	jmp	.L279
.L223:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$210, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L247:
	movss	.LC40(%rip), %xmm2
	movss	.LC30(%rip), %xmm1
	movl	.LC41(%rip), %eax
	movd	%eax, %xmm0
	call	batteryIsOk
	movl	%eax, -12(%rbp)
	movq	$57, -8(%rbp)
	jmp	.L279
.L266:
	movl	.LC42(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -56(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L279
.L210:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$225, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L270:
	cmpl	$0, -56(%rbp)
	je	.L288
	movq	$20, -8(%rbp)
	jmp	.L279
.L288:
	movq	$55, -8(%rbp)
	jmp	.L279
.L234:
	cmpl	$0, -44(%rbp)
	je	.L290
	movq	$75, -8(%rbp)
	jmp	.L279
.L290:
	movq	$48, -8(%rbp)
	jmp	.L279
.L225:
	movl	.LC22(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -76(%rbp)
	movq	$40, -8(%rbp)
	jmp	.L279
.L277:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$218, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L255:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$222, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L209:
	cmpl	$0, -92(%rbp)
	je	.L292
	movq	$36, -8(%rbp)
	jmp	.L279
.L292:
	movq	$63, -8(%rbp)
	jmp	.L279
.L275:
	movl	.LC40(%rip), %eax
	movd	%eax, %xmm0
	call	chargeRateIsAboveLimit
	movl	%eax, -44(%rbp)
	movq	$45, -8(%rbp)
	jmp	.L279
.L262:
	cmpl	$0, -80(%rbp)
	je	.L294
	movq	$56, -8(%rbp)
	jmp	.L279
.L294:
	movq	$54, -8(%rbp)
	jmp	.L279
.L254:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$211, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L257:
	movl	.LC47(%rip), %eax
	movd	%eax, %xmm0
	call	chargeRateIsAboveLimit
	movl	%eax, -36(%rbp)
	movq	$46, -8(%rbp)
	jmp	.L279
.L242:
	movl	.LC27(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -88(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L279
.L202:
	movl	.LC28(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -104(%rbp)
	movq	$5, -8(%rbp)
	jmp	.L279
.L222:
	cmpl	$0, -12(%rbp)
	je	.L296
	movq	$17, -8(%rbp)
	jmp	.L279
.L296:
	movq	$64, -8(%rbp)
	jmp	.L279
.L211:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$227, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L252:
	cmpl	$0, -108(%rbp)
	je	.L298
	movq	$35, -8(%rbp)
	jmp	.L279
.L298:
	movq	$76, -8(%rbp)
	jmp	.L279
.L267:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$212, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L269:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$203, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L265:
	movl	.LC51(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -68(%rbp)
	movq	$2, -8(%rbp)
	jmp	.L279
.L216:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$206, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC52(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L228:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$214, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC53(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L259:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$230, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC54(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L246:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$224, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC55(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L261:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$231, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC56(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L239:
	cmpl	$0, -76(%rbp)
	je	.L300
	movq	$22, -8(%rbp)
	jmp	.L279
.L300:
	movq	$24, -8(%rbp)
	jmp	.L279
.L212:
	cmpl	$0, -64(%rbp)
	je	.L302
	movq	$29, -8(%rbp)
	jmp	.L279
.L302:
	movq	$51, -8(%rbp)
	jmp	.L279
.L224:
	movl	.LC57(%rip), %eax
	movd	%eax, %xmm0
	call	chargeRateIsAboveLimit
	movl	%eax, -52(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L279
.L219:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$208, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC58(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L220:
	movss	.LC57(%rip), %xmm2
	movss	.LC59(%rip), %xmm1
	movl	.LC60(%rip), %eax
	movd	%eax, %xmm0
	call	batteryIsOk
	movl	%eax, -32(%rbp)
	movq	$30, -8(%rbp)
	jmp	.L279
.L272:
	cmpl	$0, -52(%rbp)
	je	.L304
	movq	$15, -8(%rbp)
	jmp	.L279
.L304:
	movq	$1, -8(%rbp)
	jmp	.L279
.L251:
	movl	.LC61(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -108(%rbp)
	movq	$26, -8(%rbp)
	jmp	.L279
.L240:
	movl	$1, printLanguage(%rip)
	movss	.LC62(%rip), %xmm2
	movss	.LC63(%rip), %xmm1
	movl	.LC64(%rip), %eax
	movd	%eax, %xmm0
	call	batteryIsOk
	movl	%eax, -16(%rbp)
	movq	$33, -8(%rbp)
	jmp	.L279
.L218:
	cmpl	$0, -48(%rbp)
	je	.L306
	movq	$3, -8(%rbp)
	jmp	.L279
.L306:
	movq	$58, -8(%rbp)
	jmp	.L279
.L221:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$219, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC65(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L244:
	cmpl	$0, -60(%rbp)
	je	.L308
	movq	$12, -8(%rbp)
	jmp	.L279
.L308:
	movq	$71, -8(%rbp)
	jmp	.L279
.L205:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$221, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC66(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L204:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$220, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC67(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L231:
	movl	.LC28(%rip), %eax
	movd	%eax, %xmm0
	call	chargeRateIsAboveLimit
	movl	%eax, -40(%rbp)
	movq	$42, -8(%rbp)
	jmp	.L279
.L208:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$215, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC68(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L256:
	movl	.LC69(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -72(%rbp)
	movq	$25, -8(%rbp)
	jmp	.L279
.L250:
	cmpl	$0, -28(%rbp)
	je	.L310
	movq	$69, -8(%rbp)
	jmp	.L279
.L310:
	movq	$52, -8(%rbp)
	jmp	.L279
.L226:
	movl	.LC41(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -92(%rbp)
	movq	$70, -8(%rbp)
	jmp	.L279
.L214:
	cmpl	$0, -100(%rbp)
	je	.L312
	movq	$41, -8(%rbp)
	jmp	.L279
.L312:
	movq	$73, -8(%rbp)
	jmp	.L279
.L232:
	movss	.LC24(%rip), %xmm2
	movss	.LC70(%rip), %xmm1
	movl	.LC71(%rip), %eax
	movd	%eax, %xmm0
	call	batteryIsOk
	movl	%eax, -20(%rbp)
	movq	$43, -8(%rbp)
	jmp	.L279
.L206:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$204, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC72(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L235:
	movl	.LC73(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -100(%rbp)
	movq	$65, -8(%rbp)
	jmp	.L279
.L273:
	cmpl	$0, -104(%rbp)
	je	.L314
	movq	$44, -8(%rbp)
	jmp	.L279
.L314:
	movq	$9, -8(%rbp)
	jmp	.L279
.L207:
	movl	.LC74(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -80(%rbp)
	movq	$16, -8(%rbp)
	jmp	.L279
.L245:
	cmpl	$0, -16(%rbp)
	je	.L316
	movq	$31, -8(%rbp)
	jmp	.L279
.L316:
	movq	$19, -8(%rbp)
	jmp	.L279
.L241:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$226, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC75(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L215:
	movl	$0, %eax
	jmp	.L332
.L238:
	movl	.LC76(%rip), %eax
	movd	%eax, %xmm0
	call	temperatureIsInRange
	movl	%eax, -96(%rbp)
	movq	$66, -8(%rbp)
	jmp	.L279
.L268:
	pxor	%xmm2, %xmm2
	movss	.LC77(%rip), %xmm1
	movl	.LC51(%rip), %eax
	movd	%eax, %xmm0
	call	batteryIsOk
	movl	%eax, -28(%rbp)
	movq	$28, -8(%rbp)
	jmp	.L279
.L237:
	cmpl	$0, -40(%rbp)
	je	.L319
	movq	$21, -8(%rbp)
	jmp	.L279
.L319:
	movq	$74, -8(%rbp)
	jmp	.L279
.L278:
	cmpl	$0, -88(%rbp)
	je	.L321
	movq	$49, -8(%rbp)
	jmp	.L279
.L321:
	movq	$14, -8(%rbp)
	jmp	.L279
.L233:
	cmpl	$0, -36(%rbp)
	je	.L323
	movq	$59, -8(%rbp)
	jmp	.L279
.L323:
	movq	$23, -8(%rbp)
	jmp	.L279
.L213:
	cmpl	$0, -96(%rbp)
	je	.L325
	movq	$53, -8(%rbp)
	jmp	.L279
.L325:
	movq	$7, -8(%rbp)
	jmp	.L279
.L271:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$205, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC78(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L243:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$202, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC79(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L249:
	movl	.LC23(%rip), %eax
	movd	%eax, %xmm0
	call	stateOfChargeIsInRange
	movl	%eax, -60(%rbp)
	movq	$34, -8(%rbp)
	jmp	.L279
.L236:
	cmpl	$0, -20(%rbp)
	je	.L327
	movq	$38, -8(%rbp)
	jmp	.L279
.L327:
	movq	$68, -8(%rbp)
	jmp	.L279
.L276:
	cmpl	$0, -68(%rbp)
	je	.L329
	movq	$18, -8(%rbp)
	jmp	.L279
.L329:
	movq	$62, -8(%rbp)
	jmp	.L279
.L258:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rcx
	movl	$216, %edx
	leaq	.LC36(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC80(%rip), %rax
	movq	%rax, %rdi
	call	__assert_fail@PLT
.L333:
	nop
.L279:
	jmp	.L331
.L332:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	main, .-main
	.section	.rodata
	.align 4
.LC20:
	.long	1103101952
	.align 4
.LC21:
	.long	1117257728
	.align 4
.LC22:
	.long	1101004800
	.align 4
.LC23:
	.long	1117782016
	.align 4
.LC24:
	.long	1061326684
	.align 4
.LC25:
	.long	1061997773
	.align 4
.LC26:
	.long	1110114304
	.align 4
.LC27:
	.long	1110704128
	.align 4
.LC28:
	.long	0
	.align 4
.LC29:
	.long	1074790400
	.align 4
.LC30:
	.long	1117768909
	.align 4
.LC31:
	.long	1110730342
	.align 4
.LC32:
	.long	1061494456
	.align 4
.LC33:
	.long	1117650944
	.align 4
.LC34:
	.long	1110441984
	.align 4
.LC40:
	.long	1063675494
	.align 4
.LC41:
	.long	1110677914
	.align 4
.LC42:
	.long	1117795123
	.align 4
.LC47:
	.long	-1082130432
	.align 4
.LC51:
	.long	1112014848
	.align 4
.LC57:
	.long	1060320051
	.align 4
.LC59:
	.long	1116471296
	.align 4
.LC60:
	.long	1103626240
	.align 4
.LC61:
	.long	-1110651699
	.align 4
.LC62:
	.long	1061158912
	.align 4
.LC63:
	.long	1103049523
	.align 4
.LC64:
	.long	1074580685
	.align 4
.LC69:
	.long	1101057229
	.align 4
.LC70:
	.long	1101529088
	.align 4
.LC71:
	.long	1065353216
	.align 4
.LC73:
	.long	1036831949
	.align 4
.LC74:
	.long	1100952371
	.align 4
.LC76:
	.long	1102053376
	.align 4
.LC77:
	.long	1118437376
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
