	.file	"konrad-gajdus_donut_donut_flatten.c"
	.text
	.globl	_TIG_IZ_Jy0c_envp
	.bss
	.align 8
	.type	_TIG_IZ_Jy0c_envp, @object
	.size	_TIG_IZ_Jy0c_envp, 8
_TIG_IZ_Jy0c_envp:
	.zero	8
	.globl	COLOR_CODES
	.align 32
	.type	COLOR_CODES, @object
	.size	COLOR_CODES, 96
COLOR_CODES:
	.zero	96
	.globl	LUMINANCE_CHARS
	.align 8
	.type	LUMINANCE_CHARS, @object
	.size	LUMINANCE_CHARS, 8
LUMINANCE_CHARS:
	.zero	8
	.globl	_TIG_IZ_Jy0c_argv
	.align 8
	.type	_TIG_IZ_Jy0c_argv, @object
	.size	_TIG_IZ_Jy0c_argv, 8
_TIG_IZ_Jy0c_argv:
	.zero	8
	.globl	_TIG_IZ_Jy0c_argc
	.align 4
	.type	_TIG_IZ_Jy0c_argc, @object
	.size	_TIG_IZ_Jy0c_argc, 4
_TIG_IZ_Jy0c_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"%s"
.LC11:
	.string	"%s%c\033[0m"
	.text
	.globl	render_donut
	.type	render_donut, @function
render_donut:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	leaq	-151552(%rsp), %r11
.LPSRL0:
	subq	$4096, %rsp
	orq	$0, (%rsp)
	cmpq	%r11, %rsp
	jne	.LPSRL0
	subq	$2992, %rsp
	movsd	%xmm0, -154504(%rbp)
	movsd	%xmm1, -154512(%rbp)
	movq	%rdi, -154520(%rbp)
	movsd	%xmm2, -154528(%rbp)
	movsd	%xmm3, -154536(%rbp)
	movsd	%xmm4, -154544(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -154376(%rbp)
.L74:
	cmpq	$56, -154376(%rbp)
	ja	.L85
	movq	-154376(%rbp), %rax
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
	.long	.L85-.L4
	.long	.L39-.L4
	.long	.L38-.L4
	.long	.L37-.L4
	.long	.L36-.L4
	.long	.L35-.L4
	.long	.L34-.L4
	.long	.L85-.L4
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L31-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L30-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L85-.L4
	.long	.L27-.L4
	.long	.L26-.L4
	.long	.L25-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L85-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L85-.L4
	.long	.L19-.L4
	.long	.L18-.L4
	.long	.L85-.L4
	.long	.L17-.L4
	.long	.L85-.L4
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L85-.L4
	.long	.L11-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L85-.L4
	.long	.L8-.L4
	.long	.L86-.L4
	.long	.L6-.L4
	.long	.L85-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L26:
	movl	$0, -154472(%rbp)
	movq	$53, -154376(%rbp)
	jmp	.L40
.L20:
	cmpl	$0, -154476(%rbp)
	js	.L41
	movq	$47, -154376(%rbp)
	jmp	.L40
.L41:
	movq	$13, -154376(%rbp)
	jmp	.L40
.L36:
	movq	$17, -154376(%rbp)
	jmp	.L40
.L17:
	movl	$0, -154464(%rbp)
	movq	$1, -154376(%rbp)
	jmp	.L40
.L29:
	leaq	-110016(%rbp), %rcx
	movl	-154464(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movl	-154468(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$2200, %rdx, %rdx
	addq	%rdx, %rax
	leaq	(%rcx,%rax), %rdx
	movq	-154384(%rbp), %rax
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movl	%eax, -154456(%rbp)
	movl	-154456(%rbp), %eax
	movl	%eax, -154452(%rbp)
	movl	-154452(%rbp), %eax
	cltq
	addq	%rax, -154384(%rbp)
	addl	$1, -154464(%rbp)
	movq	$1, -154376(%rbp)
	jmp	.L40
.L28:
	cmpl	$49, -154468(%rbp)
	jg	.L44
	movq	$30, -154376(%rbp)
	jmp	.L40
.L44:
	movq	$24, -154376(%rbp)
	jmp	.L40
.L3:
	cmpl	$0, -154472(%rbp)
	jns	.L46
	movq	$18, -154376(%rbp)
	jmp	.L40
.L46:
	movq	$33, -154376(%rbp)
	jmp	.L40
.L33:
	cmpl	$109, -154480(%rbp)
	jg	.L48
	movq	$25, -154376(%rbp)
	jmp	.L40
.L48:
	movq	$13, -154376(%rbp)
	jmp	.L40
.L39:
	cmpl	$109, -154464(%rbp)
	jg	.L50
	movq	$14, -154376(%rbp)
	jmp	.L40
.L50:
	movq	$46, -154376(%rbp)
	jmp	.L40
.L22:
	addl	$1, -154488(%rbp)
	movq	$40, -154376(%rbp)
	jmp	.L40
.L37:
	movl	-154480(%rbp), %eax
	movslq	%eax, %rdx
	movl	-154476(%rbp), %eax
	cltq
	imulq	$110, %rax, %rax
	addq	%rdx, %rax
	movsd	-154016(%rbp,%rax,8), %xmm1
	movsd	-154408(%rbp), %xmm0
	comisd	%xmm1, %xmm0
	jbe	.L81
	movq	$51, -154376(%rbp)
	jmp	.L40
.L81:
	movq	$13, -154376(%rbp)
	jmp	.L40
.L21:
	movq	-154384(%rbp), %rax
	movb	$0, (%rax)
	movq	$52, -154376(%rbp)
	jmp	.L40
.L23:
	movq	-154440(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154032(%rbp)
	movsd	-154032(%rbp), %xmm0
	movsd	%xmm0, -154432(%rbp)
	movq	-154440(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154024(%rbp)
	movsd	-154024(%rbp), %xmm0
	movsd	%xmm0, -154424(%rbp)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -154416(%rbp)
	movq	$10, -154376(%rbp)
	jmp	.L40
.L32:
	movsd	.LC2(%rip), %xmm0
	comisd	-154440(%rbp), %xmm0
	jbe	.L82
	movq	$21, -154376(%rbp)
	jmp	.L40
.L82:
	movq	$6, -154376(%rbp)
	jmp	.L40
.L30:
	movsd	-154416(%rbp), %xmm1
	movsd	.LC3(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -154416(%rbp)
	movq	$10, -154376(%rbp)
	jmp	.L40
.L8:
	movl	-154480(%rbp), %eax
	movslq	%eax, %rdx
	movl	-154476(%rbp), %eax
	cltq
	imulq	$110, %rax, %rax
	addq	%rdx, %rax
	movsd	-154408(%rbp), %xmm0
	movsd	%xmm0, -154016(%rbp,%rax,8)
	movsd	.LC4(%rip), %xmm0
	movq	-154400(%rbp), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -154360(%rbp)
	movsd	-154360(%rbp), %xmm1
	movsd	.LC5(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -154472(%rbp)
	movq	$56, -154376(%rbp)
	jmp	.L40
.L25:
	movsd	-154544(%rbp), %xmm1
	movsd	.LC6(%rip), %xmm0
	mulsd	%xmm0, %xmm1
	movsd	.LC7(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	movsd	-154528(%rbp), %xmm1
	movapd	%xmm1, %xmm2
	addsd	-154536(%rbp), %xmm2
	movsd	.LC8(%rip), %xmm1
	mulsd	%xmm2, %xmm1
	divsd	%xmm1, %xmm0
	movsd	%xmm0, -154448(%rbp)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -154440(%rbp)
	movq	$9, -154376(%rbp)
	jmp	.L40
.L16:
	cmpl	$109, -154484(%rbp)
	jg	.L58
	movq	$28, -154376(%rbp)
	jmp	.L40
.L58:
	movq	$23, -154376(%rbp)
	jmp	.L40
.L27:
	leaq	-110016(%rbp), %rax
	movl	$110000, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movl	$0, -154488(%rbp)
	movq	$40, -154376(%rbp)
	jmp	.L40
.L13:
	cmpl	$49, -154488(%rbp)
	jg	.L60
	movq	$27, -154376(%rbp)
	jmp	.L40
.L60:
	movq	$19, -154376(%rbp)
	jmp	.L40
.L5:
	cmpl	$0, -154480(%rbp)
	js	.L62
	movq	$8, -154376(%rbp)
	jmp	.L40
.L62:
	movq	$13, -154376(%rbp)
	jmp	.L40
.L34:
	movq	-154520(%rbp), %rax
	movq	%rax, -154384(%rbp)
	movl	$0, -154468(%rbp)
	movq	$15, -154376(%rbp)
	jmp	.L40
.L19:
	movl	$0, -154484(%rbp)
	movq	$32, -154376(%rbp)
	jmp	.L40
.L14:
	movq	-154392(%rbp), %rax
	movl	%eax, -154472(%rbp)
	movq	$53, -154376(%rbp)
	jmp	.L40
.L18:
	leaq	-110016(%rbp), %rcx
	movl	-154484(%rbp), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	movl	-154488(%rbp), %edx
	movslq	%edx, %rdx
	imulq	$2200, %rdx, %rdx
	addq	%rdx, %rax
	addq	%rcx, %rax
	movw	$32, (%rax)
	movl	-154484(%rbp), %eax
	movslq	%eax, %rdx
	movl	-154488(%rbp), %eax
	cltq
	imulq	$110, %rax, %rax
	addq	%rdx, %rax
	movsd	.LC9(%rip), %xmm0
	movsd	%xmm0, -154016(%rbp,%rax,8)
	addl	$1, -154484(%rbp)
	movq	$32, -154376(%rbp)
	jmp	.L40
.L6:
	movsd	-154440(%rbp), %xmm0
	addsd	-154416(%rbp), %xmm0
	movq	%xmm0, %rax
	movsd	.LC2(%rip), %xmm0
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	fmod@PLT
	movq	%xmm0, %rax
	movq	%rax, -154368(%rbp)
	movsd	-154368(%rbp), %xmm0
	movsd	.LC2(%rip), %xmm2
	movapd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
	movsd	.LC10(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -154460(%rbp)
	movq	LUMINANCE_CHARS(%rip), %rdx
	movl	-154472(%rbp), %eax
	cltq
	addq	%rdx, %rax
	movzbl	(%rax), %eax
	movsbl	%al, %ecx
	movl	-154460(%rbp), %eax
	cltq
	leaq	0(,%rax,8), %rdx
	leaq	COLOR_CODES(%rip), %rax
	movq	(%rdx,%rax), %rdx
	leaq	-110016(%rbp), %rdi
	movl	-154480(%rbp), %eax
	movslq	%eax, %rsi
	movq	%rsi, %rax
	salq	$2, %rax
	addq	%rsi, %rax
	salq	$2, %rax
	movl	-154476(%rbp), %esi
	movslq	%esi, %rsi
	imulq	$2200, %rsi, %rsi
	addq	%rsi, %rax
	addq	%rdi, %rax
	leaq	.LC11(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$13, -154376(%rbp)
	jmp	.L40
.L9:
	cmpl	$49, -154476(%rbp)
	jg	.L64
	movq	$3, -154376(%rbp)
	jmp	.L40
.L64:
	movq	$13, -154376(%rbp)
	jmp	.L40
.L35:
	movl	-154472(%rbp), %eax
	cltq
	movq	%rax, -154392(%rbp)
	movq	$34, -154376(%rbp)
	jmp	.L40
.L15:
	movl	-154472(%rbp), %eax
	cmpl	$6, %eax
	jbe	.L66
	movq	$20, -154376(%rbp)
	jmp	.L40
.L66:
	movq	$5, -154376(%rbp)
	jmp	.L40
.L12:
	movsd	-154440(%rbp), %xmm1
	movsd	.LC12(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -154440(%rbp)
	movq	$9, -154376(%rbp)
	jmp	.L40
.L31:
	movsd	.LC2(%rip), %xmm0
	comisd	-154416(%rbp), %xmm0
	jbe	.L83
	movq	$43, -154376(%rbp)
	jmp	.L40
.L83:
	movq	$41, -154376(%rbp)
	jmp	.L40
.L10:
	movq	-154384(%rbp), %rax
	movq	%rax, -154352(%rbp)
	addq	$1, -154384(%rbp)
	movq	-154352(%rbp), %rax
	movb	$10, (%rax)
	addl	$1, -154468(%rbp)
	movq	$15, -154376(%rbp)
	jmp	.L40
.L11:
	movq	-154416(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154344(%rbp)
	movsd	-154344(%rbp), %xmm0
	movsd	%xmm0, -154336(%rbp)
	movq	-154416(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154328(%rbp)
	movsd	-154328(%rbp), %xmm0
	movsd	%xmm0, -154320(%rbp)
	movsd	-154528(%rbp), %xmm0
	mulsd	-154432(%rbp), %xmm0
	movsd	-154536(%rbp), %xmm1
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -154312(%rbp)
	movsd	-154528(%rbp), %xmm0
	mulsd	-154424(%rbp), %xmm0
	movsd	%xmm0, -154304(%rbp)
	movq	-154512(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154296(%rbp)
	movq	-154504(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154288(%rbp)
	movq	-154512(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154280(%rbp)
	movq	-154504(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154272(%rbp)
	movq	-154512(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154264(%rbp)
	movsd	-154296(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-154336(%rbp), %xmm1
	movsd	-154288(%rbp), %xmm0
	mulsd	-154280(%rbp), %xmm0
	mulsd	-154320(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	mulsd	-154312(%rbp), %xmm0
	movsd	-154304(%rbp), %xmm1
	mulsd	-154272(%rbp), %xmm1
	mulsd	-154264(%rbp), %xmm1
	subsd	%xmm1, %xmm0
	movsd	%xmm0, -154256(%rbp)
	movq	-154512(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154248(%rbp)
	movq	-154504(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154240(%rbp)
	movq	-154512(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154232(%rbp)
	movq	-154504(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154224(%rbp)
	movq	-154512(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154216(%rbp)
	movsd	-154248(%rbp), %xmm0
	mulsd	-154336(%rbp), %xmm0
	movsd	-154240(%rbp), %xmm1
	mulsd	-154232(%rbp), %xmm1
	mulsd	-154320(%rbp), %xmm1
	subsd	%xmm1, %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-154312(%rbp), %xmm1
	movsd	-154304(%rbp), %xmm0
	mulsd	-154224(%rbp), %xmm0
	mulsd	-154216(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -154208(%rbp)
	movq	-154504(%rbp), %rax
	movq	%rax, %xmm0
	call	cos@PLT
	movq	%xmm0, %rax
	movq	%rax, -154200(%rbp)
	movq	-154504(%rbp), %rax
	movq	%rax, %xmm0
	call	sin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154192(%rbp)
	movsd	-154200(%rbp), %xmm0
	mulsd	-154312(%rbp), %xmm0
	mulsd	-154320(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	addsd	-154544(%rbp), %xmm1
	movsd	-154304(%rbp), %xmm0
	mulsd	-154192(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -154184(%rbp)
	movsd	.LC13(%rip), %xmm0
	divsd	-154184(%rbp), %xmm0
	movsd	%xmm0, -154408(%rbp)
	movsd	-154448(%rbp), %xmm0
	mulsd	-154408(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-154256(%rbp), %xmm1
	movsd	.LC14(%rip), %xmm0
	addsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -154480(%rbp)
	movsd	-154448(%rbp), %xmm0
	mulsd	-154408(%rbp), %xmm0
	mulsd	-154208(%rbp), %xmm0
	movsd	.LC15(%rip), %xmm2
	movapd	%xmm0, %xmm1
	divsd	%xmm2, %xmm1
	movsd	.LC16(%rip), %xmm0
	subsd	%xmm1, %xmm0
	cvttsd2sil	%xmm0, %eax
	movl	%eax, -154476(%rbp)
	movsd	-154336(%rbp), %xmm0
	mulsd	-154432(%rbp), %xmm0
	movsd	%xmm0, -154176(%rbp)
	movsd	-154336(%rbp), %xmm0
	mulsd	-154424(%rbp), %xmm0
	movsd	%xmm0, -154168(%rbp)
	movsd	-154320(%rbp), %xmm0
	movsd	%xmm0, -154160(%rbp)
	movsd	.LC17(%rip), %xmm0
	movsd	%xmm0, -154152(%rbp)
	movsd	.LC18(%rip), %xmm0
	movsd	%xmm0, -154144(%rbp)
	movsd	.LC19(%rip), %xmm0
	movsd	%xmm0, -154136(%rbp)
	movsd	-154176(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-154152(%rbp), %xmm1
	movsd	-154168(%rbp), %xmm0
	mulsd	-154144(%rbp), %xmm0
	addsd	%xmm0, %xmm1
	movsd	-154160(%rbp), %xmm0
	mulsd	-154136(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -154400(%rbp)
	movsd	.LC20(%rip), %xmm0
	movsd	%xmm0, -154128(%rbp)
	movsd	-154128(%rbp), %xmm0
	movq	-154400(%rbp), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	fmax@PLT
	movq	%xmm0, %rax
	movq	%rax, -154400(%rbp)
	movsd	-154256(%rbp), %xmm0
	movq	.LC21(%rip), %xmm1
	xorpd	%xmm1, %xmm0
	movsd	%xmm0, -154120(%rbp)
	movsd	-154208(%rbp), %xmm0
	movq	.LC21(%rip), %xmm1
	xorpd	%xmm1, %xmm0
	movsd	%xmm0, -154112(%rbp)
	movsd	-154544(%rbp), %xmm0
	subsd	-154184(%rbp), %xmm0
	movsd	%xmm0, -154104(%rbp)
	movsd	-154120(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	%xmm0, %xmm1
	movsd	-154112(%rbp), %xmm0
	mulsd	%xmm0, %xmm0
	addsd	%xmm0, %xmm1
	movsd	-154104(%rbp), %xmm0
	mulsd	%xmm0, %xmm0
	addsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movq	%rax, %xmm0
	call	sqrt@PLT
	movq	%xmm0, %rax
	movq	%rax, -154096(%rbp)
	movsd	-154096(%rbp), %xmm0
	movsd	%xmm0, -154088(%rbp)
	movsd	-154120(%rbp), %xmm0
	divsd	-154088(%rbp), %xmm0
	movsd	%xmm0, -154120(%rbp)
	movsd	-154112(%rbp), %xmm0
	divsd	-154088(%rbp), %xmm0
	movsd	%xmm0, -154112(%rbp)
	movsd	-154104(%rbp), %xmm0
	divsd	-154088(%rbp), %xmm0
	movsd	%xmm0, -154104(%rbp)
	movsd	-154400(%rbp), %xmm0
	addsd	%xmm0, %xmm0
	mulsd	-154176(%rbp), %xmm0
	subsd	-154152(%rbp), %xmm0
	movsd	%xmm0, -154080(%rbp)
	movsd	-154400(%rbp), %xmm0
	addsd	%xmm0, %xmm0
	mulsd	-154168(%rbp), %xmm0
	subsd	-154144(%rbp), %xmm0
	movsd	%xmm0, -154072(%rbp)
	movsd	-154400(%rbp), %xmm0
	addsd	%xmm0, %xmm0
	mulsd	-154160(%rbp), %xmm0
	subsd	-154136(%rbp), %xmm0
	movsd	%xmm0, -154064(%rbp)
	movsd	-154120(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-154080(%rbp), %xmm1
	movsd	-154112(%rbp), %xmm0
	mulsd	-154072(%rbp), %xmm0
	addsd	%xmm0, %xmm1
	movsd	-154104(%rbp), %xmm0
	mulsd	-154064(%rbp), %xmm0
	addsd	%xmm1, %xmm0
	movapd	%xmm0, %xmm1
	movq	.LC1(%rip), %rax
	movq	%rax, %xmm0
	call	fmax@PLT
	movq	%xmm0, %rax
	movq	%rax, -154056(%rbp)
	movsd	.LC22(%rip), %xmm0
	movq	-154056(%rbp), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	pow@PLT
	movq	%xmm0, %rax
	movq	%rax, -154048(%rbp)
	movsd	-154048(%rbp), %xmm0
	movsd	%xmm0, -154040(%rbp)
	movsd	-154040(%rbp), %xmm1
	movsd	.LC23(%rip), %xmm0
	mulsd	%xmm1, %xmm0
	addsd	-154400(%rbp), %xmm0
	movq	.LC13(%rip), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	fmin@PLT
	movq	%xmm0, %rax
	movq	%rax, -154400(%rbp)
	movq	$2, -154376(%rbp)
	jmp	.L40
.L38:
	movsd	-154400(%rbp), %xmm0
	pxor	%xmm1, %xmm1
	comisd	%xmm1, %xmm0
	jbe	.L84
	movq	$55, -154376(%rbp)
	jmp	.L40
.L84:
	movq	$13, -154376(%rbp)
	jmp	.L40
.L24:
	movq	$6, -154392(%rbp)
	movq	$34, -154376(%rbp)
	jmp	.L40
.L85:
	nop
.L40:
	jmp	.L74
.L86:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L76
	call	__stack_chk_fail@PLT
.L76:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	render_donut, .-render_donut
	.section	.rodata
.LC24:
	.string	"\033[2J\033[H"
	.text
	.globl	clear_screen
	.type	clear_screen, @function
clear_screen:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$1, -8(%rbp)
.L92:
	cmpq	$0, -8(%rbp)
	je	.L93
	cmpq	$1, -8(%rbp)
	jne	.L94
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -8(%rbp)
	jmp	.L90
.L94:
	nop
.L90:
	jmp	.L92
.L93:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	clear_screen, .-clear_screen
	.section	.rodata
.LC25:
	.string	" .:-=+*#%@"
.LC26:
	.string	"\033[31m"
.LC27:
	.string	"\033[32m"
.LC28:
	.string	"\033[33m"
.LC29:
	.string	"\033[34m"
.LC30:
	.string	"\033[35m"
.LC31:
	.string	"\033[36m"
.LC32:
	.string	"\033[91m"
.LC33:
	.string	"\033[92m"
.LC34:
	.string	"\033[93m"
.LC35:
	.string	"\033[94m"
.LC36:
	.string	"\033[95m"
.LC37:
	.string	"\033[96m"
	.align 8
.LC40:
	.string	"Failed to allocate memory for buffer\n"
	.align 8
.LC42:
	.string	"Tube Radius: %.2f, Donut Radius: %.2f, Viewer Distance: %.2f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	leaq	.LC25(%rip), %rax
	movq	%rax, LUMINANCE_CHARS(%rip)
	nop
.L96:
	leaq	.LC26(%rip), %rax
	movq	%rax, COLOR_CODES(%rip)
	leaq	.LC27(%rip), %rax
	movq	%rax, 8+COLOR_CODES(%rip)
	leaq	.LC28(%rip), %rax
	movq	%rax, 16+COLOR_CODES(%rip)
	leaq	.LC29(%rip), %rax
	movq	%rax, 24+COLOR_CODES(%rip)
	leaq	.LC30(%rip), %rax
	movq	%rax, 32+COLOR_CODES(%rip)
	leaq	.LC31(%rip), %rax
	movq	%rax, 40+COLOR_CODES(%rip)
	leaq	.LC32(%rip), %rax
	movq	%rax, 48+COLOR_CODES(%rip)
	leaq	.LC33(%rip), %rax
	movq	%rax, 56+COLOR_CODES(%rip)
	leaq	.LC34(%rip), %rax
	movq	%rax, 64+COLOR_CODES(%rip)
	leaq	.LC35(%rip), %rax
	movq	%rax, 72+COLOR_CODES(%rip)
	leaq	.LC36(%rip), %rax
	movq	%rax, 80+COLOR_CODES(%rip)
	leaq	.LC37(%rip), %rax
	movq	%rax, 88+COLOR_CODES(%rip)
	nop
.L97:
	movq	$0, _TIG_IZ_Jy0c_envp(%rip)
	nop
.L98:
	movq	$0, _TIG_IZ_Jy0c_argv(%rip)
	nop
.L99:
	movl	$0, _TIG_IZ_Jy0c_argc(%rip)
	nop
	nop
.L100:
.L101:
#APP
# 120 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-Jy0c--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_Jy0c_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_Jy0c_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_Jy0c_envp(%rip)
	nop
	movq	$26, -24(%rbp)
.L142:
	cmpq	$33, -24(%rbp)
	ja	.L143
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L104(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L104(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L104:
	.long	.L143-.L104
	.long	.L124-.L104
	.long	.L143-.L104
	.long	.L123-.L104
	.long	.L122-.L104
	.long	.L143-.L104
	.long	.L121-.L104
	.long	.L143-.L104
	.long	.L143-.L104
	.long	.L120-.L104
	.long	.L119-.L104
	.long	.L118-.L104
	.long	.L143-.L104
	.long	.L143-.L104
	.long	.L143-.L104
	.long	.L117-.L104
	.long	.L116-.L104
	.long	.L115-.L104
	.long	.L114-.L104
	.long	.L113-.L104
	.long	.L143-.L104
	.long	.L112-.L104
	.long	.L111-.L104
	.long	.L110-.L104
	.long	.L109-.L104
	.long	.L143-.L104
	.long	.L108-.L104
	.long	.L107-.L104
	.long	.L143-.L104
	.long	.L143-.L104
	.long	.L106-.L104
	.long	.L105-.L104
	.long	.L143-.L104
	.long	.L103-.L104
	.text
.L114:
	movsd	.LC13(%rip), %xmm0
	movsd	%xmm0, -48(%rbp)
	movsd	.LC15(%rip), %xmm0
	movsd	%xmm0, -40(%rbp)
	movsd	.LC38(%rip), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L122:
	movsd	-32(%rbp), %xmm1
	movsd	.LC17(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L106:
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -64(%rbp)
	pxor	%xmm0, %xmm0
	movsd	%xmm0, -56(%rbp)
	movsd	.LC13(%rip), %xmm0
	movsd	%xmm0, -48(%rbp)
	movsd	.LC15(%rip), %xmm0
	movsd	%xmm0, -40(%rbp)
	movsd	.LC38(%rip), %xmm0
	movsd	%xmm0, -32(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L125
.L117:
	movl	$0, %eax
	jmp	.L126
.L105:
	movl	$1, %eax
	jmp	.L126
.L124:
	movl	$30000, %edi
	call	usleep@PLT
	movsd	-64(%rbp), %xmm1
	movsd	.LC39(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -64(%rbp)
	movsd	-56(%rbp), %xmm1
	movsd	.LC12(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -56(%rbp)
	movq	$19, -24(%rbp)
	jmp	.L125
.L110:
	movq	stderr(%rip), %rax
	movq	%rax, %rcx
	movl	$37, %edx
	movl	$1, %esi
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$31, -24(%rbp)
	jmp	.L125
.L123:
	movsd	-48(%rbp), %xmm0
	movsd	.LC41(%rip), %xmm1
	subsd	%xmm1, %xmm0
	movq	.LC41(%rip), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	fmax@PLT
	movq	%xmm0, %rax
	movq	%rax, -48(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L116:
	movq	$1, -24(%rbp)
	jmp	.L125
.L109:
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$15, -24(%rbp)
	jmp	.L125
.L112:
	movsd	-32(%rbp), %xmm0
	movsd	.LC17(%rip), %xmm1
	subsd	%xmm1, %xmm0
	movq	.LC7(%rip), %rax
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	fmax@PLT
	movq	%xmm0, %rax
	movq	%rax, -32(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L108:
	movq	$33, -24(%rbp)
	jmp	.L125
.L118:
	cmpl	$0, -80(%rbp)
	je	.L127
	movq	$27, -24(%rbp)
	jmp	.L125
.L127:
	movq	$1, -24(%rbp)
	jmp	.L125
.L120:
	movsd	-40(%rbp), %xmm1
	movsd	.LC41(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -40(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L113:
	movsd	-32(%rbp), %xmm3
	movsd	-40(%rbp), %xmm2
	movsd	-48(%rbp), %xmm1
	movq	-72(%rbp), %rdx
	movsd	-56(%rbp), %xmm0
	movq	-64(%rbp), %rax
	movapd	%xmm3, %xmm4
	movapd	%xmm2, %xmm3
	movapd	%xmm1, %xmm2
	movq	%rdx, %rdi
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	call	render_donut
	call	clear_screen
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	stdout(%rip), %rcx
	movq	-8(%rbp), %rdx
	movq	-72(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movsd	-32(%rbp), %xmm1
	movsd	-40(%rbp), %xmm0
	movq	-48(%rbp), %rax
	movapd	%xmm1, %xmm2
	movapd	%xmm0, %xmm1
	movq	%rax, %xmm0
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	movl	$3, %eax
	call	printf@PLT
	call	print_controls
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	call	kbhit
	movl	%eax, -80(%rbp)
	movq	$11, -24(%rbp)
	jmp	.L125
.L115:
	movsd	-40(%rbp), %xmm0
	movsd	.LC41(%rip), %xmm1
	movapd	%xmm0, %xmm2
	subsd	%xmm1, %xmm2
	movsd	-48(%rbp), %xmm1
	movsd	.LC41(%rip), %xmm0
	addsd	%xmm0, %xmm1
	movq	%xmm1, %rax
	movapd	%xmm2, %xmm1
	movq	%rax, %xmm0
	call	fmax@PLT
	movq	%xmm0, %rax
	movq	%rax, -40(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L121:
	cmpq	$0, -72(%rbp)
	jne	.L129
	movq	$23, -24(%rbp)
	jmp	.L125
.L129:
	movq	$30, -24(%rbp)
	jmp	.L125
.L107:
	call	getchar@PLT
	movl	%eax, -76(%rbp)
	movl	-76(%rbp), %eax
	movb	%al, -81(%rbp)
	movq	$10, -24(%rbp)
	jmp	.L125
.L111:
	movsd	-48(%rbp), %xmm1
	movsd	.LC41(%rip), %xmm0
	addsd	%xmm1, %xmm0
	movsd	%xmm0, -48(%rbp)
	movq	$1, -24(%rbp)
	jmp	.L125
.L103:
	movl	$110051, %edi
	call	malloc@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -72(%rbp)
	movq	$6, -24(%rbp)
	jmp	.L125
.L119:
	movsbl	-81(%rbp), %eax
	subl	$65, %eax
	cmpl	$55, %eax
	ja	.L131
	movl	%eax, %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L133(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L133(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L133:
	.long	.L140-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L139-.L133
	.long	.L138-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L137-.L133
	.long	.L136-.L133
	.long	.L135-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L134-.L133
	.long	.L132-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L140-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L139-.L133
	.long	.L138-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L137-.L133
	.long	.L136-.L133
	.long	.L135-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L131-.L133
	.long	.L134-.L133
	.long	.L132-.L133
	.text
.L132:
	movq	$24, -24(%rbp)
	jmp	.L141
.L136:
	movq	$18, -24(%rbp)
	jmp	.L141
.L138:
	movq	$4, -24(%rbp)
	jmp	.L141
.L137:
	movq	$21, -24(%rbp)
	jmp	.L141
.L139:
	movq	$9, -24(%rbp)
	jmp	.L141
.L140:
	movq	$17, -24(%rbp)
	jmp	.L141
.L135:
	movq	$3, -24(%rbp)
	jmp	.L141
.L134:
	movq	$22, -24(%rbp)
	jmp	.L141
.L131:
	movq	$16, -24(%rbp)
	nop
.L141:
	jmp	.L125
.L143:
	nop
.L125:
	jmp	.L142
.L126:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	main, .-main
	.globl	kbhit
	.type	kbhit, @function
kbhit:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$160, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -152(%rbp)
.L157:
	cmpq	$5, -152(%rbp)
	ja	.L160
	movq	-152(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L147(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L147(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L147:
	.long	.L152-.L147
	.long	.L151-.L147
	.long	.L150-.L147
	.long	.L149-.L147
	.long	.L148-.L147
	.long	.L146-.L147
	.text
.L148:
	leaq	-144(%rbp), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	tcgetattr@PLT
	movq	-144(%rbp), %rax
	movq	-136(%rbp), %rdx
	movq	%rax, -80(%rbp)
	movq	%rdx, -72(%rbp)
	movq	-128(%rbp), %rax
	movq	-120(%rbp), %rdx
	movq	%rax, -64(%rbp)
	movq	%rdx, -56(%rbp)
	movq	-112(%rbp), %rax
	movq	-104(%rbp), %rdx
	movq	%rax, -48(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-96(%rbp), %rax
	movq	%rax, -32(%rbp)
	movl	-88(%rbp), %eax
	movl	%eax, -24(%rbp)
	movl	-68(%rbp), %eax
	andl	$-11, %eax
	movl	%eax, -68(%rbp)
	leaq	-80(%rbp), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movl	$0, %edx
	movl	$3, %esi
	movl	$0, %edi
	movl	$0, %eax
	call	fcntl@PLT
	movl	%eax, -156(%rbp)
	movl	-156(%rbp), %eax
	orb	$8, %ah
	movl	%eax, %edx
	movl	$4, %esi
	movl	$0, %edi
	movl	$0, %eax
	call	fcntl@PLT
	call	getchar@PLT
	movl	%eax, -160(%rbp)
	leaq	-144(%rbp), %rax
	movq	%rax, %rdx
	movl	$0, %esi
	movl	$0, %edi
	call	tcsetattr@PLT
	movl	-156(%rbp), %eax
	movl	%eax, %edx
	movl	$4, %esi
	movl	$0, %edi
	movl	$0, %eax
	call	fcntl@PLT
	movq	$0, -152(%rbp)
	jmp	.L153
.L151:
	movl	$1, %eax
	jmp	.L158
.L149:
	movq	stdin(%rip), %rdx
	movl	-160(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	ungetc@PLT
	movq	$1, -152(%rbp)
	jmp	.L153
.L146:
	movl	$0, %eax
	jmp	.L158
.L152:
	cmpl	$-1, -160(%rbp)
	je	.L155
	movq	$3, -152(%rbp)
	jmp	.L153
.L155:
	movq	$5, -152(%rbp)
	jmp	.L153
.L150:
	movq	$4, -152(%rbp)
	jmp	.L153
.L160:
	nop
.L153:
	jmp	.L157
.L158:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L159
	call	__stack_chk_fail@PLT
.L159:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	kbhit, .-kbhit
	.section	.rodata
.LC43:
	.string	"\033[%d;0H"
.LC44:
	.string	"Controls:"
	.align 8
.LC45:
	.string	"W/S: +/- Tube radius | A/D: +/- Donut radius | Q/E: +/- Viewer distance | R: Reset | X: Exit"
	.text
	.globl	print_controls
	.type	print_controls, @function
print_controls:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L167:
	cmpq	$2, -8(%rbp)
	je	.L162
	cmpq	$2, -8(%rbp)
	ja	.L168
	cmpq	$0, -8(%rbp)
	je	.L169
	cmpq	$1, -8(%rbp)
	jne	.L168
	movl	$52, %esi
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	.LC44(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC45(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$0, -8(%rbp)
	jmp	.L165
.L162:
	movq	$1, -8(%rbp)
	jmp	.L165
.L168:
	nop
.L165:
	jmp	.L167
.L169:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	print_controls, .-print_controls
	.section	.rodata
	.align 8
.LC1:
	.long	0
	.long	0
	.align 8
.LC2:
	.long	1413754136
	.long	1075388923
	.align 8
.LC3:
	.long	1202590843
	.long	1065646817
	.align 8
.LC4:
	.long	-858993459
	.long	1072483532
	.align 8
.LC5:
	.long	0
	.long	1075314688
	.align 8
.LC6:
	.long	0
	.long	1079738368
	.align 8
.LC7:
	.long	0
	.long	1074266112
	.align 8
.LC8:
	.long	0
	.long	1075838976
	.align 8
.LC9:
	.long	0
	.long	-1043477147
	.align 8
.LC10:
	.long	0
	.long	1076363264
	.align 8
.LC12:
	.long	1202590843
	.long	1066695393
	.align 8
.LC13:
	.long	0
	.long	1072693248
	.align 8
.LC14:
	.long	0
	.long	1078689792
	.align 8
.LC15:
	.long	0
	.long	1073741824
	.align 8
.LC16:
	.long	0
	.long	1077477376
	.align 8
.LC17:
	.long	0
	.long	1071644672
	.align 8
.LC18:
	.long	0
	.long	-1075838976
	.align 8
.LC19:
	.long	0
	.long	-1074790400
	.align 8
.LC20:
	.long	-1717986918
	.long	1070176665
	.align 16
.LC21:
	.long	0
	.long	-2147483648
	.long	0
	.long	0
	.align 8
.LC22:
	.long	0
	.long	1077149696
	.align 8
.LC23:
	.long	858993459
	.long	1071854387
	.align 8
.LC38:
	.long	0
	.long	1075052544
	.align 8
.LC39:
	.long	1202590843
	.long	1067743969
	.align 8
.LC41:
	.long	-1717986918
	.long	1069128089
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
