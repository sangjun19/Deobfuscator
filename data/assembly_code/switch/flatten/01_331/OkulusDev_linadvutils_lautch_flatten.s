	.file	"OkulusDev_linadvutils_lautch_flatten.c"
	.text
	.globl	_TIG_IZ_b4hB_argv
	.bss
	.align 8
	.type	_TIG_IZ_b4hB_argv, @object
	.size	_TIG_IZ_b4hB_argv, 8
_TIG_IZ_b4hB_argv:
	.zero	8
	.globl	_TIG_IZ_b4hB_argc
	.align 4
	.type	_TIG_IZ_b4hB_argc, @object
	.size	_TIG_IZ_b4hB_argc, 4
_TIG_IZ_b4hB_argc:
	.zero	4
	.globl	_TIG_IZ_b4hB_envp
	.align 8
	.type	_TIG_IZ_b4hB_envp, @object
	.size	_TIG_IZ_b4hB_envp, 8
_TIG_IZ_b4hB_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	".jpg"
.LC1:
	.string	"\033[35m"
.LC2:
	.string	"\033[33m"
.LC3:
	.string	"\033[0m"
.LC4:
	.string	".png"
.LC5:
	.string	"\033[36m"
.LC6:
	.string	".txt"
.LC7:
	.string	".c"
.LC8:
	.string	"\033[32m"
.LC9:
	.string	".h"
.LC10:
	.string	".gif"
	.text
	.globl	get_color_code
	.type	get_color_code, @function
get_color_code:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movq	$0, -16(%rbp)
.L42:
	cmpq	$21, -16(%rbp)
	ja	.L43
	movq	-16(%rbp), %rax
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
	.long	.L25-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L7:
	movq	-24(%rbp), %rax
	leaq	.LC0(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -48(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L26
.L21:
	leaq	.LC1(%rip), %rax
	jmp	.L27
.L11:
	leaq	.LC1(%rip), %rax
	jmp	.L27
.L10:
	leaq	.LC2(%rip), %rax
	jmp	.L27
.L13:
	leaq	.LC1(%rip), %rax
	jmp	.L27
.L17:
	movq	-56(%rbp), %rax
	movl	$46, %esi
	movq	%rax, %rdi
	call	strrchr@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L26
.L24:
	cmpl	$0, -40(%rbp)
	jne	.L28
	movq	$14, -16(%rbp)
	jmp	.L26
.L28:
	movq	$3, -16(%rbp)
	jmp	.L26
.L22:
	leaq	.LC3(%rip), %rax
	jmp	.L27
.L9:
	cmpl	$0, -36(%rbp)
	jne	.L30
	movq	$15, -16(%rbp)
	jmp	.L26
.L30:
	movq	$18, -16(%rbp)
	jmp	.L26
.L3:
	cmpl	$0, -32(%rbp)
	jne	.L32
	movq	$17, -16(%rbp)
	jmp	.L26
.L32:
	movq	$6, -16(%rbp)
	jmp	.L26
.L14:
	cmpl	$0, -48(%rbp)
	jne	.L34
	movq	$12, -16(%rbp)
	jmp	.L26
.L34:
	movq	$19, -16(%rbp)
	jmp	.L26
.L16:
	cmpl	$0, -28(%rbp)
	jne	.L36
	movq	$7, -16(%rbp)
	jmp	.L26
.L36:
	movq	$2, -16(%rbp)
	jmp	.L26
.L12:
	cmpq	$0, -24(%rbp)
	je	.L38
	movq	$10, -16(%rbp)
	jmp	.L26
.L38:
	movq	$3, -16(%rbp)
	jmp	.L26
.L6:
	movq	-24(%rbp), %rax
	leaq	.LC4(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -44(%rbp)
	movq	$5, -16(%rbp)
	jmp	.L26
.L8:
	leaq	.LC5(%rip), %rax
	jmp	.L27
.L19:
	movq	-24(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -36(%rbp)
	movq	$16, -16(%rbp)
	jmp	.L26
.L20:
	cmpl	$0, -44(%rbp)
	jne	.L40
	movq	$4, -16(%rbp)
	jmp	.L26
.L40:
	movq	$20, -16(%rbp)
	jmp	.L26
.L15:
	movq	-24(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -28(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L26
.L25:
	movq	$8, -16(%rbp)
	jmp	.L26
.L18:
	leaq	.LC8(%rip), %rax
	jmp	.L27
.L23:
	movq	-24(%rbp), %rax
	leaq	.LC9(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -32(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L26
.L5:
	movq	-24(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -40(%rbp)
	movq	$1, -16(%rbp)
	jmp	.L26
.L43:
	nop
.L26:
	jmp	.L42
.L27:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	get_color_code, .-get_color_code
	.section	.rodata
.LC11:
	.string	"-"
.LC12:
	.string	"w"
.LC13:
	.string	"Updated "
.LC14:
	.string	"x"
.LC15:
	.string	"r"
.LC16:
	.string	"%s%s%s\n"
.LC17:
	.string	"Created "
.LC18:
	.string	"Failed to create file"
.LC19:
	.string	"File: %s\n"
.LC20:
	.string	"Size: %ld bytes\n"
.LC21:
	.string	"Permissions: "
.LC22:
	.string	"cv"
.LC23:
	.string	"Failed to modify timestamps"
.LC24:
	.string	"Usage: %s [-c] [-v] filename\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$336, %rsp
	movl	%edi, -308(%rbp)
	movq	%rsi, -320(%rbp)
	movq	%rdx, -328(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_b4hB_envp(%rip)
	nop
.L45:
	movq	$0, _TIG_IZ_b4hB_argv(%rip)
	nop
.L46:
	movl	$0, _TIG_IZ_b4hB_argc(%rip)
	nop
	nop
.L47:
.L48:
#APP
# 110 "OkulusDev_linadvutils_lautch.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-b4hB--0
# 0 "" 2
#NO_APP
	movl	-308(%rbp), %eax
	movl	%eax, _TIG_IZ_b4hB_argc(%rip)
	movq	-320(%rbp), %rax
	movq	%rax, _TIG_IZ_b4hB_argv(%rip)
	movq	-328(%rbp), %rax
	movq	%rax, _TIG_IZ_b4hB_envp(%rip)
	nop
	movq	$32, -192(%rbp)
.L163:
	cmpq	$80, -192(%rbp)
	ja	.L118
	movq	-192(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L51(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L51(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L51:
	.long	.L118-.L51
	.long	.L117-.L51
	.long	.L116-.L51
	.long	.L115-.L51
	.long	.L114-.L51
	.long	.L113-.L51
	.long	.L112-.L51
	.long	.L111-.L51
	.long	.L110-.L51
	.long	.L109-.L51
	.long	.L108-.L51
	.long	.L118-.L51
	.long	.L107-.L51
	.long	.L106-.L51
	.long	.L105-.L51
	.long	.L104-.L51
	.long	.L103-.L51
	.long	.L118-.L51
	.long	.L102-.L51
	.long	.L101-.L51
	.long	.L118-.L51
	.long	.L100-.L51
	.long	.L99-.L51
	.long	.L98-.L51
	.long	.L118-.L51
	.long	.L97-.L51
	.long	.L96-.L51
	.long	.L95-.L51
	.long	.L94-.L51
	.long	.L93-.L51
	.long	.L92-.L51
	.long	.L91-.L51
	.long	.L90-.L51
	.long	.L89-.L51
	.long	.L118-.L51
	.long	.L88-.L51
	.long	.L87-.L51
	.long	.L86-.L51
	.long	.L85-.L51
	.long	.L118-.L51
	.long	.L118-.L51
	.long	.L84-.L51
	.long	.L83-.L51
	.long	.L82-.L51
	.long	.L81-.L51
	.long	.L80-.L51
	.long	.L79-.L51
	.long	.L78-.L51
	.long	.L77-.L51
	.long	.L76-.L51
	.long	.L75-.L51
	.long	.L74-.L51
	.long	.L118-.L51
	.long	.L118-.L51
	.long	.L118-.L51
	.long	.L118-.L51
	.long	.L73-.L51
	.long	.L72-.L51
	.long	.L71-.L51
	.long	.L70-.L51
	.long	.L118-.L51
	.long	.L69-.L51
	.long	.L68-.L51
	.long	.L67-.L51
	.long	.L66-.L51
	.long	.L65-.L51
	.long	.L64-.L51
	.long	.L63-.L51
	.long	.L62-.L51
	.long	.L61-.L51
	.long	.L60-.L51
	.long	.L59-.L51
	.long	.L118-.L51
	.long	.L58-.L51
	.long	.L57-.L51
	.long	.L56-.L51
	.long	.L55-.L51
	.long	.L54-.L51
	.long	.L53-.L51
	.long	.L52-.L51
	.long	.L50-.L51
	.text
.L102:
	leaq	.LC11(%rip), %rax
	movq	%rax, -256(%rbp)
	movq	$9, -192(%rbp)
	jmp	.L118
.L75:
	leaq	.LC12(%rip), %rax
	movq	%rax, -208(%rbp)
	movq	$63, -192(%rbp)
	jmp	.L118
.L50:
	leaq	.LC11(%rip), %rax
	movq	%rax, -240(%rbp)
	movq	$4, -192(%rbp)
	jmp	.L118
.L97:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -192(%rbp)
	jmp	.L118
.L76:
	movl	-136(%rbp), %eax
	andl	$64, %eax
	testl	%eax, %eax
	je	.L119
	movq	$8, -192(%rbp)
	jmp	.L118
.L119:
	movq	$78, -192(%rbp)
	jmp	.L118
.L114:
	movq	-240(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$7, -192(%rbp)
	jmp	.L118
.L92:
	movl	-136(%rbp), %eax
	andl	$32, %eax
	testl	%eax, %eax
	je	.L121
	movq	$41, -192(%rbp)
	jmp	.L118
.L121:
	movq	$80, -192(%rbp)
	jmp	.L118
.L68:
	cmpb	$0, -295(%rbp)
	je	.L123
	movq	$33, -192(%rbp)
	jmp	.L118
.L123:
	movq	$22, -192(%rbp)
	jmp	.L118
.L105:
	movb	$0, -296(%rbp)
	movb	$1, -295(%rbp)
	movb	$0, -294(%rbp)
	movq	$0, -272(%rbp)
	movq	$37, -192(%rbp)
	jmp	.L118
.L104:
	leaq	.LC14(%rip), %rax
	movq	%rax, -200(%rbp)
	movq	$44, -192(%rbp)
	jmp	.L118
.L73:
	leaq	.LC15(%rip), %rax
	movq	%rax, -264(%rbp)
	movq	$71, -192(%rbp)
	jmp	.L118
.L52:
	movq	-232(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$45, -192(%rbp)
	jmp	.L118
.L91:
	movq	-248(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$30, -192(%rbp)
	jmp	.L118
.L107:
	movq	-272(%rbp), %rax
	movq	%rax, %rdi
	call	get_color_code
	movq	%rax, -184(%rbp)
	movq	-272(%rbp), %rdx
	movq	-184(%rbp), %rax
	leaq	.LC3(%rip), %rcx
	movq	%rax, %rsi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$3, -192(%rbp)
	jmp	.L118
.L61:
	leaq	.LC14(%rip), %rax
	movq	%rax, -224(%rbp)
	movq	$59, -192(%rbp)
	jmp	.L118
.L110:
	leaq	.LC14(%rip), %rax
	movq	%rax, -248(%rbp)
	movq	$31, -192(%rbp)
	jmp	.L118
.L80:
	movl	-136(%rbp), %eax
	andl	$8, %eax
	testl	%eax, %eax
	je	.L125
	movq	$69, -192(%rbp)
	jmp	.L118
.L125:
	movq	$6, -192(%rbp)
	jmp	.L118
.L53:
	leaq	.LC11(%rip), %rax
	movq	%rax, -248(%rbp)
	movq	$31, -192(%rbp)
	jmp	.L118
.L117:
	cmpb	$0, -293(%rbp)
	je	.L127
	movq	$21, -192(%rbp)
	jmp	.L118
.L127:
	movq	$43, -192(%rbp)
	jmp	.L118
.L98:
	movl	-136(%rbp), %eax
	andl	$256, %eax
	testl	%eax, %eax
	je	.L129
	movq	$56, -192(%rbp)
	jmp	.L118
.L129:
	movq	$10, -192(%rbp)
	jmp	.L118
.L54:
	leaq	-160(%rbp), %rdx
	movq	-272(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	stat@PLT
	movl	%eax, -280(%rbp)
	cmpl	$0, -280(%rbp)
	sete	%al
	movb	%al, -293(%rbp)
	movq	$62, -192(%rbp)
	jmp	.L118
.L60:
	movl	-136(%rbp), %eax
	andl	$128, %eax
	testl	%eax, %eax
	je	.L131
	movq	$67, -192(%rbp)
	jmp	.L118
.L131:
	movq	$18, -192(%rbp)
	jmp	.L118
.L115:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L164
	jmp	.L165
.L103:
	cmpl	$-1, -288(%rbp)
	jne	.L134
	movq	$74, -192(%rbp)
	jmp	.L118
.L134:
	movq	$19, -192(%rbp)
	jmp	.L118
.L100:
	movq	-88(%rbp), %rax
	movq	%rax, -176(%rbp)
	movl	$0, %edi
	call	time@PLT
	movq	%rax, -168(%rbp)
	leaq	-176(%rbp), %rdx
	movq	-272(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	utime@PLT
	movl	%eax, -284(%rbp)
	movq	$46, -192(%rbp)
	jmp	.L118
.L87:
	movq	-272(%rbp), %rax
	movl	$420, %edx
	movl	$64, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open@PLT
	movl	%eax, -276(%rbp)
	movl	-276(%rbp), %eax
	movl	%eax, -288(%rbp)
	movq	$16, -192(%rbp)
	jmp	.L118
.L55:
	movl	-136(%rbp), %eax
	andl	$1, %eax
	testl	%eax, %eax
	je	.L136
	movq	$15, -192(%rbp)
	jmp	.L118
.L136:
	movq	$61, -192(%rbp)
	jmp	.L118
.L72:
	leaq	.LC11(%rip), %rax
	movq	%rax, -208(%rbp)
	movq	$63, -192(%rbp)
	jmp	.L118
.L62:
	leaq	.LC11(%rip), %rax
	movq	%rax, -232(%rbp)
	movq	$79, -192(%rbp)
	jmp	.L118
.L96:
	cmpb	$0, -293(%rbp)
	je	.L138
	movq	$25, -192(%rbp)
	jmp	.L118
.L138:
	movq	$27, -192(%rbp)
	jmp	.L118
.L109:
	movq	-256(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$49, -192(%rbp)
	jmp	.L118
.L106:
	movl	-136(%rbp), %eax
	andl	$2, %eax
	testl	%eax, %eax
	je	.L140
	movq	$50, -192(%rbp)
	jmp	.L118
.L140:
	movq	$57, -192(%rbp)
	jmp	.L118
.L67:
	movq	-208(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$76, -192(%rbp)
	jmp	.L118
.L74:
	cmpl	$99, -292(%rbp)
	je	.L142
	cmpl	$118, -292(%rbp)
	jne	.L143
	movq	$73, -192(%rbp)
	jmp	.L144
.L142:
	movq	$58, -192(%rbp)
	jmp	.L144
.L143:
	movq	$29, -192(%rbp)
	nop
.L144:
	jmp	.L118
.L101:
	movl	-288(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$22, -192(%rbp)
	jmp	.L118
.L90:
	movq	$14, -192(%rbp)
	jmp	.L118
.L63:
	leaq	.LC12(%rip), %rax
	movq	%rax, -256(%rbp)
	movq	$9, -192(%rbp)
	jmp	.L118
.L70:
	movq	-224(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$75, -192(%rbp)
	jmp	.L118
.L112:
	leaq	.LC11(%rip), %rax
	movq	%rax, -224(%rbp)
	movq	$59, -192(%rbp)
	jmp	.L118
.L95:
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -192(%rbp)
	jmp	.L118
.L85:
	movl	optind(%rip), %eax
	cmpl	%eax, -308(%rbp)
	jle	.L145
	movq	$65, -192(%rbp)
	jmp	.L118
.L145:
	movq	$35, -192(%rbp)
	jmp	.L118
.L69:
	leaq	.LC11(%rip), %rax
	movq	%rax, -200(%rbp)
	movq	$44, -192(%rbp)
	jmp	.L118
.L71:
	movb	$1, -295(%rbp)
	movq	$37, -192(%rbp)
	jmp	.L118
.L57:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L56:
	movl	-136(%rbp), %eax
	andl	$4, %eax
	testl	%eax, %eax
	je	.L147
	movq	$66, -192(%rbp)
	jmp	.L118
.L147:
	movq	$42, -192(%rbp)
	jmp	.L118
.L77:
	cmpb	$0, -293(%rbp)
	je	.L149
	movq	$5, -192(%rbp)
	jmp	.L118
.L149:
	movq	$26, -192(%rbp)
	jmp	.L118
.L59:
	movq	-264(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$70, -192(%rbp)
	jmp	.L118
.L99:
	cmpb	$0, -294(%rbp)
	je	.L151
	movq	$1, -192(%rbp)
	jmp	.L118
.L151:
	movq	$43, -192(%rbp)
	jmp	.L118
.L94:
	cmpl	$-1, -292(%rbp)
	je	.L153
	movq	$51, -192(%rbp)
	jmp	.L118
.L153:
	movq	$38, -192(%rbp)
	jmp	.L118
.L65:
	movl	optind(%rip), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	movq	-320(%rbp), %rax
	addq	%rdx, %rax
	movq	(%rax), %rax
	movq	%rax, -272(%rbp)
	movq	$77, -192(%rbp)
	jmp	.L118
.L78:
	leaq	.LC12(%rip), %rax
	movq	%rax, -232(%rbp)
	movq	$79, -192(%rbp)
	jmp	.L118
.L58:
	movb	$1, -296(%rbp)
	movq	$37, -192(%rbp)
	jmp	.L118
.L81:
	movq	-200(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$10, %edi
	call	putchar@PLT
	movq	$26, -192(%rbp)
	jmp	.L118
.L113:
	movq	-272(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC19(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-112(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC20(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -192(%rbp)
	jmp	.L118
.L89:
	movzbl	-293(%rbp), %eax
	xorl	$1, %eax
	testb	%al, %al
	je	.L155
	movq	$36, -192(%rbp)
	jmp	.L118
.L155:
	movq	$22, -192(%rbp)
	jmp	.L118
.L86:
	movq	-320(%rbp), %rcx
	movl	-308(%rbp), %eax
	leaq	.LC22(%rip), %rdx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	getopt@PLT
	movl	%eax, -292(%rbp)
	movq	$28, -192(%rbp)
	jmp	.L118
.L66:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L84:
	leaq	.LC15(%rip), %rax
	movq	%rax, -240(%rbp)
	movq	$4, -192(%rbp)
	jmp	.L118
.L108:
	leaq	.LC11(%rip), %rax
	movq	%rax, -264(%rbp)
	movq	$71, -192(%rbp)
	jmp	.L118
.L83:
	leaq	.LC11(%rip), %rax
	movq	%rax, -216(%rbp)
	movq	$2, -192(%rbp)
	jmp	.L118
.L79:
	cmpl	$-1, -284(%rbp)
	jne	.L157
	movq	$64, -192(%rbp)
	jmp	.L118
.L157:
	movq	$43, -192(%rbp)
	jmp	.L118
.L64:
	leaq	.LC15(%rip), %rax
	movq	%rax, -216(%rbp)
	movq	$2, -192(%rbp)
	jmp	.L118
.L111:
	movl	-136(%rbp), %eax
	andl	$16, %eax
	testl	%eax, %eax
	je	.L159
	movq	$47, -192(%rbp)
	jmp	.L118
.L159:
	movq	$68, -192(%rbp)
	jmp	.L118
.L88:
	movq	-320(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC24(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L93:
	movq	-320(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC24(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L82:
	cmpb	$0, -296(%rbp)
	je	.L161
	movq	$48, -192(%rbp)
	jmp	.L118
.L161:
	movq	$26, -192(%rbp)
	jmp	.L118
.L116:
	movq	-216(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$13, -192(%rbp)
	nop
.L118:
	jmp	.L163
.L165:
	call	__stack_chk_fail@PLT
.L164:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
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
